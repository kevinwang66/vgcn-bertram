from collections import Counter
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import scipy.sparse as sp

import numpy as np
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from transformers.activations import get_activation
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import BertModel, BertConfig
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.utils import (
    logging,
)

from .configuration_VGCNBert import VGCNBertConfig
# from configuration_VGCNBert import VGCNBertConfig

logger = logging.get_logger(__name__)

def _normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))  # D-degree matrix
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def _scipy_to_torch(sparse):
    sparse = sparse.tocoo() if sparse.getformat() != "coo" else sparse
    i = torch.LongTensor(np.vstack((sparse.row, sparse.col)))
    v = torch.from_numpy(sparse.data)
    return torch.sparse_coo_tensor(i, v, torch.Size(sparse.shape)).coalesce()


def _delete_special_terms(words: list, terms: set):
    return set([w for w in words if w not in terms])


def _build_pmi_graph(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    window_size=20,
    algorithm="npmi",
    edge_threshold=0.0,
    remove_stopwords=False,
    min_freq_to_keep=2,
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Build statistical word graph from text samples using PMI or NPMI algorithm.
    """

    # Tokenize the text samples. The tokenizer should be same as that in the combined Bert-like model.
    # Remove stopwords and special terms
    # Get vocabulary and the word frequency
    words_to_remove = (set({"[CLS]", "[SEP]"})
    )
    vocab_counter = Counter()
    texts_words = []
    for t in texts:
        words = tokenizer.tokenize(t)
        words = _delete_special_terms(words, words_to_remove)
        if len(words) > 0:
            vocab_counter.update(Counter(words))
            texts_words.append(words)

    # Set [PAD] as the head of vocabulary
    # Remove word with freq<n and re generate texts
    new_vocab_counter = Counter({"[PAD]": 0})
    new_vocab_counter.update(
        Counter({k: v for k, v in vocab_counter.items() if v >= min_freq_to_keep})
        if min_freq_to_keep > 1
        else vocab_counter
    )
    vocab_counter = new_vocab_counter

    # Generate new texts by removing words with freq<n
    if min_freq_to_keep > 1:
        texts_words = [list(filter(lambda w: vocab_counter[w] >= min_freq_to_keep, words)) for words in texts_words]
    texts = [" ".join(words).strip() for words in texts_words if len(words) > 0]

    vocab_size = len(vocab_counter)
    vocab = list(vocab_counter.keys())
    assert vocab[0] == "[PAD]"
    vocab_indices = {k: i for i, k in enumerate(vocab)}

    # Get the pieces from sliding windows
    windows = []
    for t in texts:
        words = t.split()
        word_ids = [vocab_indices[w] for w in words]
        length = len(word_ids)
        if length <= window_size:
            windows.append(word_ids)
        else:
            for j in range(length - window_size + 1):
                word_ids = word_ids[j : j + window_size]
                windows.append(word_ids)

    # Get the window-count that every word appeared (count 1 for the same window).
    # Get window-count that every word-pair appeared (count 1 for the same window).
    vocab_window_counter = Counter()
    word_pair_window_counter = Counter()
    for word_ids in windows:
        word_ids = list(set(word_ids))
        vocab_window_counter.update(Counter(word_ids))
        word_pair_window_counter.update(
            Counter(
                [
                    f(i, j)
                    # (word_ids[i], word_ids[j])
                    for i in range(1, len(word_ids))
                    for j in range(i)
                    # adding inverse pair
                    for f in (lambda x, y: (word_ids[x], word_ids[y]), lambda x, y: (word_ids[y], word_ids[x]))
                ]
            )
        )

    # Calculate NPMI
    vocab_adj_row = []
    vocab_adj_col = []
    vocab_adj_weight = []

    total_windows = len(windows)
    for wid_pair in word_pair_window_counter.keys():
        i, j = wid_pair
        pair_count = word_pair_window_counter[wid_pair]
        i_count = vocab_window_counter[i]
        j_count = vocab_window_counter[j]
        value = (
            (math.log(1.0 * i_count * j_count / (total_windows**2)) / math.log(1.0 * pair_count / total_windows) - 1)
            if algorithm == "npmi"
            else (math.log((1.0 * pair_count / total_windows) / (1.0 * i_count * j_count / (total_windows**2))))
        )
        if value > edge_threshold:
            vocab_adj_row.append(i)
            vocab_adj_col.append(j)
            vocab_adj_weight.append(value)

    # Build vocabulary adjacency matrix
    vocab_adj = sp.csr_matrix(
        (vocab_adj_weight, (vocab_adj_row, vocab_adj_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )
    vocab_adj.setdiag(1.0)

    # Padding the first row and column, "[PAD]" is the first word in the vocabulary.
    assert vocab_adj[0, :].sum() == 1
    assert vocab_adj[:, 0].sum() == 1
    vocab_adj[:, 0] = 0
    vocab_adj[0, :] = 0

    wgraph_id_to_tokenizer_id_map = {v: tokenizer.vocab[k] for k, v in vocab_indices.items()}
    wgraph_id_to_tokenizer_id_map = dict(sorted(wgraph_id_to_tokenizer_id_map.items()))

    return (
        vocab_adj,
        vocab_indices,
        wgraph_id_to_tokenizer_id_map,
    )

class WordGraphBuilder:
    """
    Word graph based on adjacency matrix, construct from text samples or pre-defined word-pair relations

    You may (or not) first preprocess the text before build the graph,
    e.g. Stopword removal, String cleaning, Stemming, Nomolization, Lemmatization

    Params:
        `rows`: List[str] of text samples, or pre-defined word-pair relations: List[Tuple[str, str, float]]
        `tokenizer`: The same pretrained tokenizer that is used for the model late.
        `window_size`:  Available only for statistics generation (rows is text samples).
            Size of the sliding window for collecting the pieces of text
            and further calculate the NPMI value, default is 20.
        `algorithm`:  Available only for statistics generation (rows is text samples) -- "npmi" or "pmi", default is "npmi".
        `edge_threshold`: Available only for statistics generation (rows is text samples). Graph edge value threshold, default is 0. Edge value is between -1 to 1.
        `remove_stopwords`: Build word graph with the words that are not stopwords, default is False.
        `min_freq_to_keep`: Available only for statistics generation (rows is text samples). Build word graph with the words that occurred at least n times in the corpus, default is 2.

    Properties:
        `adjacency_matrix`: scipy.sparse.csr_matrix, the word graph in sparse adjacency matrix form.
        `vocab_indices`: indices of word graph vocabulary words.
        `wgraph_id_to_tokenizer_id_map`: map from word graph vocabulary word id to tokenizer vocabulary word id.

    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        rows: list,
        tokenizer: PreTrainedTokenizerBase,
        window_size=20,
        algorithm="npmi",
        edge_threshold=0.0,
        remove_stopwords=False,
        min_freq_to_keep=2,
    ):
        (
            adjacency_matrix,
            _,
            wgraph_id_to_tokenizer_id_map,
        ) = _build_pmi_graph(
            rows, tokenizer, window_size, algorithm, edge_threshold, remove_stopwords, min_freq_to_keep
        )

        adjacency_matrix=_scipy_to_torch(_normalize_adj(adjacency_matrix)) if adjacency_matrix is not None else None
        return adjacency_matrix, wgraph_id_to_tokenizer_id_map

class VgcnParameterList(nn.ParameterList):
    def __init__(self, values=None, requires_grad=True) -> None:
        super().__init__(values)
        self.requires_grad = requires_grad

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        keys = filter(lambda x: x.startswith(prefix), state_dict.keys())
        for k in keys:
            self.append(nn.Parameter(state_dict[k], requires_grad=self.requires_grad))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        for i in range(len(self)):
            if self[i].layout is torch.sparse_coo and not self[i].is_coalesced():
                self[i] = self[i].coalesce()
            self[i].requires_grad = self.requires_grad


class VocabGraphConvolution(nn.Module):
    """Vocabulary GCN module.

    Params:
        `wgraphs`: List of vocabulary graph, normally adjacency matrix
        `wgraph_id_to_tokenizer_id_maps`: wgraph.vocabulary to tokenizer.vocabulary id-mapping
        `hid_dim`: The hidden dimension after `GCN=XAW` (GCN layer)
        `out_dim`: The output dimension after `out=Relu(XAW)W`  (GCN output)
        `activation`: The activation function in `out=act(XAW)W`
        `dropout_rate`: The dropout probabilitiy in `out=dropout(act(XAW))W`.

    Inputs:
        `X_dv`: the feature of mini batch document, can be TF-IDF (batch, vocab), or word embedding (batch, word_embedding_dim, vocab)

    Outputs:
        The graph embedding representation, dimension (batch, `out_dim`) or (batch, word_embedding_dim, `out_dim`)

    """

    def __init__(
        self,
        hid_dim: int,
        out_dim: int,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
        activation=None,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fc_hg = nn.Linear(hid_dim, out_dim)
        self.fc_hg._is_vgcn_linear = True
        self.activation = get_activation(activation) if activation else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        # TODO: add a Linear layer for vgcn fintune/pretrain task

        # after init.set_wgraphs, _init_weights will set again the mode (transparent,normal,uniform)
        # but if load wgraph parameters from checkpoint/pretrain, the mode weights will be updated from to checkpoint
        # you can call again set_parameters to change the mode
        self.set_wgraphs(wgraphs, wgraph_id_to_tokenizer_id_maps)

    def set_parameters(self, mode="transparent"):
        """Set the parameters of the model (transparent, uniform, normal)."""
        assert mode in ["transparent", "uniform", "normal"]
        for n, p in self.named_parameters():
            if n.startswith("W"):
                nn.init.constant_(p, 1.0) if mode == "transparent" else nn.init.normal_(
                    p, mean=0.0, std=0.02
                ) if mode == "normal" else nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.fc_hg.weight.data.fill_(1.0) if mode == "transparent" else self.fc_hg.weight.data.normal_(
            mean=0.0, std=0.02
        ) if mode == "normal" else nn.init.kaiming_uniform_(self.fc_hg.weight, a=math.sqrt(5))
        self.fc_hg.bias.data.zero_()

    def set_wgraphs(
        self,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
        mode="transparent",
    ):
        assert (
            wgraphs is None
            and wgraph_id_to_tokenizer_id_maps is None
            or wgraphs is not None
            and wgraph_id_to_tokenizer_id_maps is not None
        )
        self.wgraphs: VgcnParameterList = (
            self._prepare_wgraphs(wgraphs) if wgraphs else VgcnParameterList(requires_grad=False)
        )
        self.gvoc_ordered_tokenizer_id_arrays, self.tokenizer_id_to_wgraph_id_arrays = VgcnParameterList(
            requires_grad=False
        ), VgcnParameterList(requires_grad=False)
        if wgraph_id_to_tokenizer_id_maps:
            (
                self.gvoc_ordered_tokenizer_id_arrays,
                self.tokenizer_id_to_wgraph_id_arrays,
            ) = self._prepare_inverted_arrays(wgraph_id_to_tokenizer_id_maps)
        self.W_vh_list = VgcnParameterList(requires_grad=True)
        self.W_vh_list._is_vgcn_weights = True
        for g in self.wgraphs:
            self.W_vh_list.append(nn.Parameter(torch.randn(g.shape[0], self.hid_dim)))
            # self.W_vh_list.append(nn.Parameter(torch.ones(g.shape[0], self.hid_dim)))
        self.set_parameters(mode=mode)

    def _prepare_wgraphs(self, wgraphs: list) -> VgcnParameterList:
        # def _zero_padding_graph(adj_matrix: torch.Tensor):
        #     if adj_matrix.layout is not torch.sparse_coo:
        #         adj_matrix=adj_matrix.to_sparse_coo()
        #     indices=adj_matrix.indices()+1
        #     padded_adj= torch.sparse_coo_tensor(indices=indices, values=adj_matrix.values(), size=(adj_matrix.shape[0]+1,adj_matrix.shape[1]+1))
        #     return padded_adj.coalesce()
        glist = VgcnParameterList(requires_grad=False)
        for g in wgraphs:
            assert g.layout is torch.sparse_coo
            # g[0,:] and g[:,0] should be 0
            assert 0 not in g.indices()
            glist.append(nn.Parameter(g.coalesce(), requires_grad=False))
        return glist

    def _prepare_inverted_arrays(self, wgraph_id_to_tokenizer_id_maps: List[dict]):
        wgraph_id_to_tokenizer_id_maps = [dict(sorted(m.items())) for m in wgraph_id_to_tokenizer_id_maps]
        assert all([list(m.keys())[-1] == len(m) - 1 for m in wgraph_id_to_tokenizer_id_maps])
        gvoc_ordered_tokenizer_id_arrays = VgcnParameterList(
            [
                nn.Parameter(torch.LongTensor(list(m.values())), requires_grad=False)
                for m in wgraph_id_to_tokenizer_id_maps
            ],
            requires_grad=False,
        )

        tokenizer_id_to_wgraph_id_arrays = VgcnParameterList(
            [
                nn.Parameter(torch.zeros(max(m.values()) + 1, dtype=torch.long), requires_grad=False)
                for m in wgraph_id_to_tokenizer_id_maps
            ],
            requires_grad=False,
        )
        for m, t in zip(wgraph_id_to_tokenizer_id_maps, tokenizer_id_to_wgraph_id_arrays):
            for graph_id, tok_id in m.items():
                t[tok_id] = graph_id

        return gvoc_ordered_tokenizer_id_arrays, tokenizer_id_to_wgraph_id_arrays

    def get_subgraphs(self, adj_matrix: torch.Tensor, gx_ids: torch.LongTensor):
        device = gx_ids.device
        batch_size = gx_ids.shape[0]
        batch_masks = torch.any(
            torch.any(
                (adj_matrix.indices().view(-1) == gx_ids.unsqueeze(-1)).view(batch_size, gx_ids.shape[1], 2, -1), dim=1
            ),
            dim=1,
        )
        nnz_len = len(adj_matrix.values())

        batch_values = adj_matrix.values().unsqueeze(0).repeat(batch_size, 1)
        batch_values = batch_values.view(-1)[batch_masks.view(-1)]

        batch_positions = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, nnz_len)
        indices = torch.cat([batch_positions.view(1, -1), adj_matrix.indices().repeat(1, batch_size)], dim=0)
        indices = indices[batch_masks.view(-1).expand(3, -1)].view(3, -1)

        batch_sub_adj_matrix = torch.sparse_coo_tensor(
            indices=indices,
            values=batch_values.view(-1),
            size=(batch_size, adj_matrix.size(0), adj_matrix.size(1)),
            dtype=adj_matrix.dtype,
            device=device,
        )

        return batch_sub_adj_matrix.coalesce()

    def forward(self, word_embeddings: nn.Embedding, input_ids: torch.Tensor):  # , position_ids: torch.Tensor = None):
        if not self.wgraphs:
            raise ValueError(
                "No wgraphs is provided. There are 3 ways to initalize wgraphs:"
                " instantiate VGCN_BERT with wgraphs, or call model.vgcn_bert.set_wgraphs(),"
                " or load from_pretrained/checkpoint (make sure there is wgraphs in checkpoint"
                " or you should call set_wgraphs)."
            )
        device = input_ids.device
        batch_size = input_ids.shape[0]
        word_emb_dim = word_embeddings.weight.shape[1]

        gx_ids_list = []
        # positon_embeddings_in_gvocab_order_list=[]
        for m in self.tokenizer_id_to_wgraph_id_arrays:
            # tmp_ids is still in sentence order, but value is graph id, e.g. [0, 5, 2, 2, 0, 10,0]
            # 0 means no correspond graph id (like padding in graph), so we need to replace it with 0
            tmp_ids = input_ids.clone()
            m = m.to(device)
            tmp_ids = tmp_ids.to(device)
            tmp_ids[tmp_ids > len(m) - 1] = 0
            tmp_ids = m[tmp_ids]

            # # position in graph is meaningless and computationally expensive
            # if position_ids:
            #     position_ids_in_g=torch.zeros(g.shape[0], dtype=torch.LongTensor)
            #     # maybe gcn_swop_eye in original vgcn_bert preprocess is more efficient?
            #     for p_id, g_id in zip(position_ids, tmp_ids):
            #         position_ids_in_g[g_id]=p_id
            #     position_embeddings_in_g=self.position_embeddings(position_ids_in_g)
            #     position_embeddings_in_g*=position_ids_in_g>0
            #     positon_embeddings_in_gvocab_order_list.append(position_embeddings_in_g)

            gx_ids_list.append(torch.unique(tmp_ids, dim=1))

        # G_embedding=(act(V1*A1_sub*W1_vh)+act(V2*A2_sub*W2_vh)）*W_hg
        fused_H = torch.zeros((batch_size, word_emb_dim, self.hid_dim), device=device)
        for gv_ids, g, gx_ids, W_vh in zip(  # , position_in_gvocab_ev
            self.gvoc_ordered_tokenizer_id_arrays,
            self.wgraphs,
            gx_ids_list,
            self.W_vh_list,
            # positon_embeddings_in_gvocab_order_list,
        ):
            # batch_A1_sub*W1_vh, batch_A2_sub*W2_vh, ...
            gv_ids = gv_ids.to(device)
            g = g.to(device)
            W_vh = W_vh.to(device)
            gx_ids = gx_ids.to(device)

            sub_wgraphs = self.get_subgraphs(g, gx_ids)
            H_vh = torch.bmm(sub_wgraphs, W_vh.unsqueeze(0).expand(batch_size, *W_vh.shape))

            # V1*batch_A1_sub*W1_vh, V2*batch_A2_sub*W2_vh, ...
            gvocab_ev = word_embeddings(gv_ids).t()
            # if position_ids:
            #     gvocab_ev += position_in_gvocab_ev
            H_eh = gvocab_ev.matmul(H_vh)

            # fc -> act -> dropout
            if self.activation:
                H_eh = self.activation(H_eh)
            if self.dropout:
                H_eh = self.dropout(H_eh)

            fused_H += H_eh

        # fused_H=LayerNorm(fused_H) # embedding assemble layer will do LayerNorm
        out_ge = self.fc_hg(fused_H).transpose(1, 2)
        # self.dropout(out_ge) # embedding assemble layer will do dropout
        return out_ge


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #

def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)


def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()

class VGCNEmbeddings(nn.Module):
    """Construct the embeddings from word, VGCN graph, position and token_type embeddings."""

    def __init__(
        self,
        config: PretrainedConfig,
        wgraphs: Optional[list] = None,
        wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)

        self.vgcn_graph_embds_dim = config.vgcn_graph_embds_dim
        self.vgcn = VocabGraphConvolution(
            hid_dim=config.vgcn_hidden_dim,
            out_dim=config.vgcn_graph_embds_dim,
            wgraphs=wgraphs,
            wgraph_id_to_tokenizer_id_maps=wgraph_id_to_tokenizer_id_maps,
            activation=config.vgcn_activation,
            dropout_rate=config.vgcn_dropout,
        )

        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
                input_ids is mandatory in vgcn-bert.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """

        # input_ids is mandatory in vgcn-bert
        input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        # device = input_embeds.device
        # input_lengths = (
        #     (input_ids > 0).sum(-1)
        #     if input_ids is not None
        #     else torch.ones(input_embeds.size(0), device=device, dtype=torch.int64) * input_embeds.size(1)
        # )

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)

        if self.vgcn_graph_embds_dim > 0:
            graph_embeds = self.vgcn(self.word_embeddings, input_ids)  # , position_ids)

            # vgcn_words_embeddings = input_embeds.clone()
            # for i in range(self.vgcn_graph_embds_dim):
            #     tmp_pos = (input_lengths - 2 - self.vgcn_graph_embds_dim + 1 + i) + torch.arange(
            #         0, input_embeds.shape[0]
            #     ).to(device) * input_embeds.shape[1]
            #     vgcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos, :] = graph_embeds[:, :, i]

            embeddings = torch.cat([embeddings, graph_embeds], dim=1)  # (bs, max_seq_length+graph_emb_dim_size, dim)

        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings

# Copied from transformers.models.distilbert.modeling_distilbert.DistilBertModel with DISTILBERT->VGCNBERT,DistilBert->VGCNBert
# class VGCNBertModel(nn.Module):
#     def __init__(
#         self,
#         config: PretrainedConfig,
#         wgraphs: Optional[list] = None,
#         wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
#     ):
#         super().__init__(config)
#
#         self.embeddings = VGCNEmbeddings(config, wgraphs, wgraph_id_to_tokenizer_id_maps)  # Graph Embeddings
#         self.transformer = Transformer(config)  # Encoder
#
#         self.wgraph_builder = WordGraphBuilder()
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#     def set_wgraphs(
#         self,
#         wgraphs: Optional[list] = None,
#         wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
#         mode="transparent",
#     ):
#         self.embeddings.vgcn.set_wgraphs(wgraphs, wgraph_id_to_tokenizer_id_maps, mode)
#
#     def get_position_embeddings(self) -> nn.Embedding:
#         """
#         Returns the position embeddings
#         """
#         return self.embeddings.position_embeddings
#
#     def resize_position_embeddings(self, new_num_position_embeddings: int):
#         """
#         Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
#
#         Arguments:
#             new_num_position_embeddings (`int`):
#                 The number of new position embedding matrix. If position embeddings are learned, increasing the size
#                 will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
#                 end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
#                 size will add correct vectors at the end following the position encoding algorithm, whereas reducing
#                 the size will remove vectors from the end.
#         """
#         num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings
#
#         # no resizing needs to be done if the length stays the same
#         if num_position_embeds_diff == 0:
#             return
#
#         logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
#         self.config.max_position_embeddings = new_num_position_embeddings
#
#         old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()
#
#         self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)
#
#         if self.config.sinusoidal_pos_embds:
#             create_sinusoidal_embeddings(
#                 n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
#             )
#         else:
#             with torch.no_grad():
#                 if num_position_embeds_diff > 0:
#                     self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
#                         old_position_embeddings_weight
#                     )
#                 else:
#                     self.embeddings.position_embeddings.weight = nn.Parameter(
#                         old_position_embeddings_weight[:num_position_embeds_diff]
#                     )
#         # move position_embeddings to correct device
#         self.embeddings.position_embeddings.to(self.device)
#
#     def get_input_embeddings(self) -> nn.Embedding:
#         return self.embeddings.word_embeddings
#
#     def set_input_embeddings(self, new_embeddings: nn.Embedding):
#         self.embeddings.word_embeddings = new_embeddings
#
#     def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]]):
#         """
#         Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
#         class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.transformer.layer[layer].attention.prune_heads(heads)
#
#     @add_start_docstrings_to_model_forward(VGCNBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=BaseModelOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")
#
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#         if attention_mask is None:
#             attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
#
#         # Prepare head mask if needed
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
#
#         embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)
#
#         if self.embeddings.vgcn_graph_embds_dim > 0:
#             attention_mask = torch.cat(
#                 [attention_mask, torch.ones((input_shape[0], self.embeddings.vgcn_graph_embds_dim), device=device)],
#                 dim=1,
#             )
#
#         return self.transformer(
#             x=embeddings,
#             attn_mask=attention_mask,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

class VGCNBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VGCNBertConfig
    load_tf_weights = None
    base_model_prefix = "vgcn_bert"

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if getattr(module, "_is_vgcn_linear", False):
                if self.config.vgcn_weight_init_mode == "transparent":
                    module.weight.data.fill_(1.0)
                elif self.config.vgcn_weight_init_mode == "normal":
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                elif self.config.vgcn_weight_init_mode == "uniform":
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                else:
                    raise ValueError(f"Unknown VGCN-BERT weight init mode: {self.config.vgcn_weight_init_mode}.")
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.ParameterList):
            if getattr(module, "_is_vgcn_weights", False):
                for p in module:
                    if self.config.vgcn_weight_init_mode == "transparent":
                        nn.init.constant_(p, 1.0)
                    elif self.config.vgcn_weight_init_mode == "normal":
                        nn.init.normal_(p, mean=0.0, std=self.config.initializer_range)
                    elif self.config.vgcn_weight_init_mode == "uniform":
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    else:
                        raise ValueError(f"Unknown VGCN-BERT weight init mode: {self.config.vgcn_weight_init_mode}.")

class VGCNBertModel(PreTrainedModel):
    def __init__(
            self,
            config: PretrainedConfig,
            wgraphs: Optional[list] = None,
            wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
    ):
        super().__init__()
        self.config=config
        self.embeddings = VGCNEmbeddings(config, wgraphs, wgraph_id_to_tokenizer_id_maps)

        self.encoder = BertModel(config).encoder

        self.wgraph_builder = WordGraphBuilder()

        # self.post_init()

    def set_wgraphs(
            self,
            wgraphs: Optional[list] = None,
            wgraph_id_to_tokenizer_id_maps: Optional[List[dict]] = None,
            mode="transparent",
    ):
        self.embeddings.vgcn.set_wgraphs(wgraphs, wgraph_id_to_tokenizer_id_maps, mode)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        返回位置嵌入
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        调整位置嵌入的大小，如果 `new_num_position_embeddings != config.max_position_embeddings`。
        """
        # 与 BERT 原始实现保持一致
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        if num_position_embeds_diff == 0:
            return

        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()

        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)

        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
            )
        else:
            with torch.no_grad():
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
                        old_position_embeddings_weight
                    )
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )
        self.embeddings.position_embeddings.to(self.device)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:

        # Ensure default values for output options
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Check input parameters
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Ensure attention_mask is created if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Using VGCN Embeddings to get embeddings
        embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)

        # Update attention_mask to match the shape of embeddings
        if self.embeddings.vgcn_graph_embds_dim > 0:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((input_shape[0], self.embeddings.vgcn_graph_embds_dim), device=device)],
                dim=1,
            )
        #     # Debugging prints
        #     print(f"Input IDs shape: {input_ids.shape}")  # Should be (8, 128)
        #     print(f"Attention Mask shape: {attention_mask.shape}")  # Should be (8, 144)
        #     print(f"Embeddings shape: {embeddings.shape}")  # Should be (8, 144, 768)
        #     print(f"Updated Attention Mask shape: {attention_mask.shape}")  # Should be (8, 144)
        #
        # # Pass to the BERT encoder part
        # encoder_outputs = self.encoder(
        #     embeddings,
        #     attention_mask=attention_mask,
        #     head_mask=head_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        #
        # return encoder_outputs

        # Convert attention_mask to the correct shape expected by BertEncoder
        extended_attention_mask = attention_mask[:, None, None, :]  # (bs, 1, 1, seq_length+graph_emb_dim)
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)  # Ensure compatibility with embeddings dtype
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # Apply large negative mask for padding positions

        # Debugging prints
        # print(f"Input IDs shape: {input_ids.shape}")
        # print(f"Attention Mask shape: {attention_mask.shape}")
        # print(f"Embeddings shape: {embeddings.shape}")
        # print(f"Updated Extended Attention Mask shape: {extended_attention_mask.shape}")

        # Pass to the BERT encoder part
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs