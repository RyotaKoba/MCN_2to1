from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
os.environ['CUDA_DEVICE_ORDER'] ="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
from args import get_args
import torch as th
th.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from youcook_dataloader import Youcook_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from model_kmeans_ICCV import Net
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from metrics import compute_metrics, print_computed_metrics
from gensim.models.keyedvectors import KeyedVectors

args = get_args()

we = None
if args.tri_modal or not args.random_audio_windows:
    print('Loading word vectors: {}'.format(args.word2vec_path))
    we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    print('Load word vectors done')
if args.eval_youcook:
    dataset_val = Youcook_DataLoader(
        data=args.youcook_val_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.youcook_num_frames_multiplier,
        tri_modal=args.tri_modal,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_msrvtt:
    msrvtt_testset = MSRVTT_DataLoader(
        data_path=args.msrvtt_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=False,
        tri_modal=args.tri_modal,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
net = Net(
    embd_dim=args.embd_dim,
    video_dim=args.feature_dim,
    we_dim=args.we_dim,
    tri_modal=args.tri_modal,
    tri_modal_fuse=args.tri_modal_fuse,
    cluster_size=args.cluster_size,
    layer=args.layer,
    project=args.project,
    project_dim=args.project_dim,
    multi_cluster=args.multi_cluster,
    recon=args.recon,
    withMLP=args.withMLP,
    recon_size=args.recon_size,
    cooperative=args.cooperative
    )

apex = False
net.cuda()
net = th.nn.DataParallel(net)
# net.train()

if args.pretrain_path != '' and args.apex_level == 0:
    net.module.load_checkpoint(args.pretrain_path)

def Eval_retrieval(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for data in tqdm(eval_dataloader):
            video = data['video'].cuda()
            audio = data['audio'].cuda()
            nframes = data['nframes'].cuda()
            if args.tri_modal:
                text = data['text'].cuda()
                if args.tri_modal_fuse==1: # AVLnet-Text
                    audio_text, video = model(video, audio, nframes, text)
                    m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                else:
                    if args.recon==1:
                        audio, video, text, recon_loss = model(video, audio, nframes, text)
                    else:
                        audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)

                    if args.eval_msrvtt==1:
                        audio_video=video+audio
                    else:
                        audio_video = video+audio
                    m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
            else:
                audio, video = model(video, audio, nframes)
                m = th.matmul(audio, video.t()).cpu().detach().numpy()
            task = data['task'].cuda().cpu().detach().numpy()
            metrics = compute_metrics(m, task, args.eval_lang_retrieval, args.eval_msrvtt)
            print_computed_metrics(metrics)

if args.eval_youcook:
    Eval_retrieval(net, dataloader_val, 'YouCook2')
if args.eval_msrvtt:
    Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
