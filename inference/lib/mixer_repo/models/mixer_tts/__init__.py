
from .mixer_tts import MixerTTSModel

net_config = {
    "n_mel_channels": 80,
    "num_tokens": 178,
    "padding_idx": 0,
    "symbols_embedding_dim": 384,

    "encoder_feature_dim": None,
    "encoder_kernel_sizes": [11, 13, 15, 17, 19, 21],
    "encoder_num_layers": 6,
    "encoder_expansion_factor": 4,
    "encoder_dropout": 0.15,

    "decoder_num_tokens": -1,
    "decoder_feature_dim": None,
    "decoder_kernel_sizes": [15, 17, 19, 21, 23, 25, 27, 29, 31],
    "decoder_num_layers": 9,
    "decoder_expansion_factor": 4,
    "decoder_dropout": 0.15,

    # "duration_predictor_input_size": 384,
    "duration_predictor_kernel_size": 3,
    "duration_predictor_filter_size": 256,
    "duration_predictor_dropout": 0.15,
    "duration_predictor_n_layers": 2,

    # "pitch_predictor_input_size": 384,
    "pitch_predictor_kernel_size": 3,
    "pitch_predictor_filter_size": 256,
    "pitch_predictor_dropout": 0.15,
    "pitch_predictor_n_layers": 2,
    "pitch_emb_in_channels": 1,
    # "pitch_emb_out_channels": 384,
    "pitch_emb_kernel_size": 3,
    
    "energy_conditioning": True,
    "energy_conditioning": 3,
    "energy_predictor_filter_size": 256,
    "energy_predictor_dropout": 0.15,
    "energy_predictor_n_layers": 2,
    "energy_embedding_kernel_size": 3,

    "aligner_n_text_channels": None,

    "n_speakers": 8,
    "n_emotions": 8
}