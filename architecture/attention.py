import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Generic Multi-Head Self-Attention.

    Input
    -----
    Q : (B,N,D)
    K : (B,N,D)
    V : (B,N,D)

    Output
    ------
    (B,N,D)
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

    def forward(self, Q, K, V):

        B, N, D = Q.shape

        # ----------------------------
        # Split into heads
        # ----------------------------

        Q = Q.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K = K.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V = V.view(
            B,
            N,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        # ----------------------------
        # Attention
        # ----------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores = scores * self.scale

        attn = F.softmax(
            scores,
            dim=-1
        )

        attn = self.dropout(attn)

        out = torch.matmul(
            attn,
            V
        )

        # ----------------------------
        # Merge heads
        # ----------------------------

        out = out.transpose(
            1,
            2
        )

        out = out.reshape(
            B,
            N,
            D
        )

        out = self.out_proj(out)

        return out
    
class SelfSubtract(nn.Module):
    """
    Self-Subtract Module

    Applies temporal differencing to the query
    and key projections while preserving the
    sequence length.

    Input
    -----
    (B, N, T, D)

    Output
    ------
    (B, N, T, D)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

        # Temporal differences

        diff = x[:, :, 1:, :] - x[:, :, :-1, :]

        # Preserve first frame

        first = x[:, :, :1, :]

        out = torch.cat(
            [first, diff],
            dim=2
        )

        return out

class TemporalAttention(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.self_subtract = SelfSubtract()

        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout
        )

    def forward(self,x):

        B,T,N,D = x.shape

        # --------------------
        # reshape
        # --------------------

        x = x.permute(
            0,
            2,
            1,
            3
        )

        x = x.reshape(
            B*N,
            T,
            D
        )

        # --------------------
        # projections
        # --------------------

        Q = self.q_proj(x)

        K = self.k_proj(x)

        V = self.v_proj(x)

        # --------------------
        # reshape for
        # SelfSubtract
        # --------------------

        Q = Q.reshape(
            B,
            N,
            T,
            D
        )

        K = K.reshape(
            B,
            N,
            T,
            D
        )

        # --------------------
        # ISTVT novelty
        # --------------------

        Q = self.self_subtract(Q)

        K = self.self_subtract(K)

        # --------------------
        # flatten again
        # --------------------

        Q = Q.reshape(
            B*N,
            T,
            D
        )

        K = K.reshape(
            B*N,
            T,
            D
        )

        # --------------------
        # attention
        # --------------------

        out = self.attention(
            Q,
            K,
            V
        )

        # --------------------
        # restore shape
        # --------------------

        out = out.reshape(
            B,
            N,
            T,
            D
        )

        out = out.permute(
            0,
            2,
            1,
            3
        )

        return out

class SpatialAttention(nn.Module):
    """
    Spatial Self-Attention

    Performs attention within each frame independently.

    Input
    -----
    (B,T,N,D)

    Output
    ------
    (B,T,N,D)
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.q_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.k_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.v_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout
        )

    def forward(self, x):

        B, T, N, D = x.shape

        # ------------------------
        # (B,T,N,D)
        #
        # ->
        #
        # (B*T,N,D)
        # ------------------------

        x = x.reshape(
            B * T,
            N,
            D
        )

        # ------------------------
        # Q,K,V
        # ------------------------

        Q = self.q_proj(x)

        K = self.k_proj(x)

        V = self.v_proj(x)

        # ------------------------
        # Spatial Attention
        # ------------------------

        out = self.attention(
            Q,
            K,
            V
        )

        # ------------------------
        # Restore shape
        # ------------------------

        out = out.reshape(
            B,
            T,
            N,
            D
        )

        return out

if __name__ == "__main__":

    x = torch.randn(
        2,
        6,
        65,
        512
    )

    print("Input :", x.shape)

    temporal = TemporalAttention()
    spatial = SpatialAttention()

    y = temporal(x)
    print("After Temporal:", y.shape)

    z = spatial(y)
    print("After Spatial :", z.shape)