from fedlearning.compressors.stosignsgd import SignSGDCompressor, StoSignSGDCompressor, OptimalStoSignSGDCompressor

compressor_registry = {
    "signSGD":           SignSGDCompressor,
    "sto_signSGD":       StoSignSGDCompressor,
    "optim_sto_signSGD": OptimalStoSignSGDCompressor
}
