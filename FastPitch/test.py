import torch
from nemo.collections.tts.models import HifiGanModel
hfg_path_gt = "HifiGan--val_loss=0.3817-epoch=34-last.ckpt"
vocoder_model = HifiGanModel.load_from_checkpoint(checkpoint_path=hfg_path_gt).eval().cuda()


torch.save(vocoder_model.generator.state_dict(), 'hifigan_generator.pth')
# torch.save(vocoder_model.discriminator.state_dict(), 'hifigan_discriminator.pth')