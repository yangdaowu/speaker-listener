import torch 
import torch.nn as nn
import torch.nn.functional as F

class CosSinPositionalEncoding(nn.Module):
    def __init__(self, d_model, period=100):
        super().__init__()
        self.d_model = d_model
        self.period = period
    def forward(self, seq_length, device=None):
        if device is None:
            device = torch.device('cpu')
        pe = torch.zeros(seq_length, self.d_model, device=device)
        position = torch.arange(seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / self.d_model))
        position_mod = position % self.period
        pe[:, 0::2] = torch.cos(position_mod * div_term)
        pe[:, 1::2] = torch.sin(position_mod * div_term)
        return pe

class BiasedCrossModalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, query, key, value, bias_mask=None):
        out, _ = self.mha(query, key, value, attn_mask=bias_mask)
        return out

class G_m(nn.Module):
    def __init__(self, d_model, num_heads=4, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.cross_attn = BiasedCrossModalMultiHeadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model))
        self.out_proj = nn.Linear(d_model, d_model * 2)
    def forward(self, h_t, fused_feature, bias_mask=None):
        x = h_t + fused_feature
        x = x.unsqueeze(1)
        attn_out = self.cross_attn(query=x, key=x, value=x, bias_mask=bias_mask)
        attn_out = attn_out.squeeze(1)
        h_next = h_t + self.fc(attn_out)
        out = self.out_proj(h_next)
        d = out.shape[-1] // 2
        beta = out[:, :d]
        rho = out[:, d:]
        return beta, rho, h_next

class AutoregressiveDecouplingDecoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4, hidden_dim=256, num_steps=60, dropout=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.pos_encoding = CosSinPositionalEncoding(d_model)
        self.g_m = G_m(d_model, num_heads, hidden_dim, dropout=dropout)
    def forward(self, fused_seq, init_state, bias_mask=None):
        B, T, d_model = fused_seq.shape
        beta_seq = []
        rho_seq = []
        h_t = init_state
        pos_emb = self.pos_encoding(self.num_steps, device=fused_seq.device)
        for t in range(self.num_steps):
            fused_t = fused_seq[:, t, :] + pos_emb[t].unsqueeze(0)
            beta, rho, h_t = self.g_m(h_t, fused_t, bias_mask=bias_mask)
            beta_seq.append(beta)
            rho_seq.append(rho)
        beta_seq = torch.stack(beta_seq, dim=1)
        rho_seq = torch.stack(rho_seq, dim=1)
        return beta_seq, rho_seq

class ResponsiveDialogueGenerator(nn.Module):
    def __init__(self, d_model=256, num_heads=4, hidden_dim=256, num_steps=60, out_dim_upper=50, out_dim_lower=50, dropout=0.1):
        super().__init__()
        self.dynamic_predictor = AutoregressiveDecouplingDecoder(d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim, num_steps=num_steps, dropout=dropout)
        self.upper_proj = nn.Linear(d_model, out_dim_upper)
        self.lower_proj = nn.Linear(d_model, out_dim_lower)
    def forward(self, fused_seq, init_state, bias_mask=None):
        beta_seq, rho_seq = self.dynamic_predictor(fused_seq, init_state, bias_mask=bias_mask)
        beta_pred = self.upper_proj(beta_seq)
        rho_pred = self.lower_proj(rho_seq)
        return beta_pred, rho_pred

class InputFusionModule(nn.Module):
    def __init__(self, audio_dim, speaker_img_dim, participant_img_dim, attitude_dim, d_model):
        super().__init__()
        self.audio_fc = nn.Linear(audio_dim, d_model)
        self.speaker_img_fc = nn.Linear(speaker_img_dim, d_model)
        self.participant_img_fc = nn.Linear(participant_img_dim, d_model)
        self.attitude_fc = nn.Linear(attitude_dim, d_model)
    def forward(self, audio_seq, speaker_images_seq, participant_image, attitude):
        audio_emb = self.audio_fc(audio_seq)
        speaker_img_emb = self.speaker_img_fc(speaker_images_seq)
        fused_seq = audio_emb + speaker_img_emb
        part_img_emb = self.participant_img_fc(participant_image)
        attitude_emb = self.attitude_fc(attitude)
        init_state = part_img_emb + attitude_emb
        return fused_seq, init_state

class DialogueGenerationModel(nn.Module):
    def __init__(self, audio_dim, speaker_img_dim, participant_img_dim, attitude_dim, d_model=256, num_heads=4, hidden_dim=256, num_steps=60, out_dim_upper=50, out_dim_lower=50, dropout=0.1):
        super().__init__()
        self.input_fusion = InputFusionModule(audio_dim, speaker_img_dim, participant_img_dim, attitude_dim, d_model)
        self.responsive_gen = ResponsiveDialogueGenerator(d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim, num_steps=num_steps, out_dim_upper=out_dim_upper, out_dim_lower=out_dim_lower, dropout=dropout)
    def forward(self, audio_seq, speaker_images_seq, participant_image, attitude, bias_mask=None):
        fused_seq, init_state = self.input_fusion(audio_seq, speaker_images_seq, participant_image, attitude)
        beta_pred, rho_pred = self.responsive_gen(fused_seq, init_state, bias_mask=bias_mask)
        return beta_pred, rho_pred

if __name__ == '__main__':
    B, T = 1, 60
    audio_dim = 128
    speaker_img_dim = 128
    participant_img_dim = 128
    attitude_dim = 32
    d_model = 256
    audio_seq = torch.randn(B, T, audio_dim)
    speaker_images_seq = torch.randn(B, T, speaker_img_dim)
    participant_image = torch.randn(B, participant_img_dim)
    attitude = torch.randn(B, attitude_dim)
    bias_mask = torch.zeros(B, 4, T, T)
    for i in range(T):
        for j in range(i+1, T):
            bias_mask[:, :, i, j] = float('-inf')
    model = DialogueGenerationModel(audio_dim=audio_dim, speaker_img_dim=speaker_img_dim, participant_img_dim=participant_img_dim, attitude_dim=attitude_dim, d_model=d_model, num_heads=4, hidden_dim=256, num_steps=T, out_dim_upper=50, out_dim_lower=50, dropout=0.1)
    beta_pred, rho_pred = model(audio_seq, speaker_images_seq, participant_image, attitude, bias_mask=bias_mask)
    print("beta_pred shape:", beta_pred.shape)
    print("rho_pred shape:", rho_pred.shape)
