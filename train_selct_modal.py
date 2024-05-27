from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import random
import os
import time
import csv
import pdb
import sys
import pickle
#GPU USAGE
os.environ['CUDA_DEVICE_ORDER'] ="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import numpy as np
from args import get_args
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from gensim.models.keyedvectors import KeyedVectors
import torch.nn as nn
import torch as th
th.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
import torch.optim as optim
from youtube_dataloader import Youtube_DataLoader
from youcook_dataloader import Youcook_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from model_kmeans_ICCV import Net
from loss import MMS_loss
from metrics import compute_metrics, print_computed_metrics, AverageMeter
from datetime import datetime
import math
from torch.optim.lr_scheduler import LambdaLR
from fast_pytorch_kmeans import KMeans
import torch.nn.functional as F
from model_davenet import load_DAVEnet
from torch.utils.tensorboard import SummaryWriter
from line_profiler import LineProfiler

whole_time = time.time()
#seed
random.seed(time.time())

now = datetime.now()
profile=LineProfiler()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

args = get_args()
if args.verbose:
    print(args)

print("1")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
print("2")

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

caption = None
if not(args.youcook) and not(args.msrvtt) and not(args.lsmdc):
    if not args.random_audio_windows:
        print('Loading HowTo100M captions: {}'.format(args.caption_path))
        caption = pickle.load(open(args.caption_path, 'rb'))
        print('Load HowTo100M captions done')

we = None
if args.tri_modal or not args.random_audio_windows:
    print('Loading word vectors: {}'.format(args.word2vec_path))
    we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    print('Load word vectors done')

dataset = Youtube_DataLoader(
    csv=args.train_csv,
    features_path=args.features_path,
    features_path_audio=args.features_path_audio,
    caption=caption,
    min_time=args.min_time,
    max_words=args.max_words,
    min_words=args.min_words,
    feature_framerate=args.feature_framerate,
    we=we,
    we_dim=args.we_dim,
    n_pair=args.n_pair,
    num_audio_frames=args.howto_audio_frames,
    random_audio_windows=args.random_audio_windows,
)
dataset_size = len(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_thread_reader,
    shuffle=True,
    batch_sampler=None,
    drop_last=True,
    pin_memory=True,
)

print("4")
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
# Optimizers + Loss
if args.loss == 0:
    loss_op = MMS_loss()
net.cuda()
loss_op.cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, len(dataloader) * args.epochs)
print("5")

if args.apex_level == 0:
    apex = False
elif args.apex_level == 1:
    from apex import amp, optimizers
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    apex = True
net = th.nn.DataParallel(net)
net.train()

if args.pretrain_path != '' and args.apex_level == 1:
    amp_checkpoint_path = os.path.join(os.path.dirname(args.pretrain_path), 'amp_checkpoint.pt')
    checkpoint = th.load(amp_checkpoint_path, map_location='cpu')
    net.module.load_state_dict(checkpoint['net'],strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint["scheduler"])
    amp.load_state_dict(checkpoint['amp'])
    print("Loaded AMP checkpoint")
elif args.pretrain_path != '' and args.apex_level == 0:
    net.module.load_checkpoint(args.pretrain_path)

if args.verbose:
    print('Starting training loop ...')
print("6")
def update_queue(queue,use_the_queue,fuse):
    bs = int(4096/2)
    fuse2 = fuse.detach()
    fuse2 = fuse2.view(-1, 32, fuse2.shape[-1])
    fuse2 = fuse2[:,:16,:]
    fuse2 = fuse2.reshape(-1, fuse2.shape[-1])
    out = fuse.detach()
    if queue is not None:  # no queue in first round
        if use_the_queue or not th.all(queue[ -1, :] == 0):  # queue[2,3840,128] if never use the queue or the queue is not full
            use_the_queue = True
            # print('use queue')
            out = th.cat((queue,fuse.detach()))  # queue [1920*128] w_t [128*3000] = 1920*3000 out [32*3000] 1952*3000

            #print('out size',out.shape)
        # fill the queue
        queue[ bs:] = queue[ :-bs].clone()  # move 0-6 to 1-7 place
        queue[:bs] = fuse2
    return queue,out,use_the_queue

def cluster_contrast(fushed,centroid,labels,bs):
    S = th.matmul(fushed, centroid.t())
    # print(centroid)
    target = th.zeros(bs,centroid.shape[0]).to(S.device)

    target[range(target.shape[0]), labels] = 1

    S = S - target * (0.001)

    if args.nce==0:
        karisoft = F.log_softmax(S, dim=1)
        # print("LogSoftMax結果")
        # print(S)
        # print("logSoftmax結果")
        I2C_loss = F.nll_loss(karisoft, labels)
        # I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), labels)

    else:
        S = S.view(S.shape[0], S.shape[1], -1)
        nominator = S * target[:, :, None]
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = S.view(S.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        I2C_loss = th.mean(denominator - nominator)

    return I2C_loss

def text_freeze(model):
    for param in model.module.text_pooling_caption.parameters():
        param.requires_grad = False
    for param in model.module.GU_text_captions.parameters():
        param.requires_grad = False
    for param in model.module.recon_t.parameters():
        param.requires_grad = False
        
def audio_freeze(model):
    for param in model.module.GU_audio.parameters():
        param.requires_grad = False
    for param in model.module.recon_a.parameters():
        param.requires_grad = False
    for param in model.module.DAVEnet.parameters():
        param.requires_grad = False
    for param in model.module.DAVEnet_projection.parameters():
        param.requires_grad = False

def video_freeze(model):
    for param in model.module.GU_video.parameters():
        param.requires_grad = False
    for param in model.module.recon_v.parameters():
        param.requires_grad = False

def TrainOneBatch(model, opt, data, loss_fun,queue_v,use_the_queue, scheduler, epoch,i_batch, centroid, apex=False):
    video = data['video'].cuda(non_blocking=True)
    audio = data['audio'].cuda(non_blocking=True)
    nframes = data['nframes'].cuda(non_blocking=True)
    text = data['text'].cuda(non_blocking=True)
    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])
    nframes = nframes.view(-1)
    opt.zero_grad()
    bs = video.size(0)  # 256
    with th.set_grad_enabled(True):
        audio, video, text, recon_loss = model(video, audio, nframes, text)
        recon_w = 50
        recon_loss = th.mean(recon_loss) * recon_w
        video_out = video
        audio_out = audio
        text_out = text
        fushed = (video_out + audio_out + text_out) / 3
        sim_audio_text = th.matmul(audio, text.t())
        sim_audio_video = th.matmul(audio, video.t())
        sim_text_video = th.matmul(text, video.t())
        if args.AV_T==1:
            if epoch == 0 and i_batch == 0:
                print("AV_T active")
            if args.pretrain_path == '':
                text_freeze(model)
                loss = loss_fun(sim_audio_video)
                fushed = (video_out + audio_out) / 2
            else:
                if epoch == 0 and i_batch == 0:
                    print("AV_T add T active")
                audio_freeze(model)
                video_freeze(model)
                loss = loss_fun(sim_audio_text) + loss_fun(sim_text_video)
            loss = loss / 2
        elif args.AT_V==1:
            if epoch == 0 and i_batch == 0:
                print("AT_V active")
            if args.pretrain_path == '':
                video_freeze(model)
                loss = loss_fun(sim_audio_text)
                fushed = (audio_out + text_out) / 2
            else:
                if epoch == 0 and i_batch == 0:
                    print("AT_V add V active")
                text_freeze(model)
                audio_freeze(model)
                loss = loss_fun(sim_audio_video) + loss_fun(sim_text_video)
            loss = loss / 2
        elif args.VT_A==1:
            if epoch == 0 and i_batch == 0:
                print("VT_A active")
            if args.pretrain_path == '':
                audio_freeze(model)
                loss = loss_fun(sim_text_video)
                fushed = (text_out + video_out) / 2
            else:
                if epoch == 0 and i_batch == 0:
                    print("VT_A add A active")
                text_freeze(model)
                video_freeze(model)
                loss = loss_fun(sim_audio_text) + loss_fun(sim_audio_video)
        else:
            if epoch == 0 and i_batch == 0:
                print("3modal start")
            loss_av = loss_fun(sim_audio_video)
            loss_vt = loss_fun(sim_text_video)
            loss_at = loss_fun(sim_audio_text)
            loss = loss_av + loss_at + loss_vt

         #clustering       
        if args.kmeans==1:
            if args.use_queue==1:
                queue_v,out,use_the_queue = update_queue(queue_v,use_the_queue,fushed.detach())
                kmeans = KMeans(n_clusters=args.cluster_size, mode='cosine')#, verbose=1)

                if args.fastC==1:
                    if i_batch%(int(args.queue_size))==0:
                        labels = kmeans.fit_predict(out,centroid)
                        centroid = kmeans.centroids
                    else:
                        labels = kmeans.max_sim(a=out, b=centroid)[1]

                else:
                    labels = kmeans.fit_predict(out)
                    centroid = kmeans.centroids

            else:
                kmeans = KMeans(n_clusters=args.cluster_size, mode='cosine', verbose=1)
                labels = kmeans.fit_predict(fushed)
        
            if args.AV_T == 1:
                if args.pretrain_path == '':
                    loss_val = cluster_contrast(video_out, centroid, labels[-bs:], bs) + \
                                cluster_contrast(audio_out, centroid, labels[-bs:], bs)
                else:
                    loss_val = cluster_contrast(text_out, centroid, labels[-bs:], bs)
                loss_val = loss_val / 2
            elif args.AT_V == 1:
                if args.pretrain_path == '':
                    loss_val = cluster_contrast(text_out, centroid, labels[-bs:], bs) + \
                                cluster_contrast(audio_out, centroid, labels[-bs:], bs)
                else:
                    loss_val = cluster_contrast(video_out, centroid, labels[-bs:], bs)
                loss_val = loss_val / 2
            elif args.VT_A == 1:
                if args.pretrain_path == '':
                    loss_val = cluster_contrast(text_out, centroid, labels[-bs:], bs) + \
                                cluster_contrast(video_out, centroid, labels[-bs:], bs)
                else:
                    loss_val = cluster_contrast(audio_out, centroid, labels[-bs:], bs)
            else:
                cluster_vid=cluster_contrast(video_out, centroid,labels[-bs:],bs)
                cluster_aud=cluster_contrast(audio_out, centroid, labels[-bs:], bs)
                cluster_txt=cluster_contrast(text_out, centroid, labels[-bs:], bs)
                loss_val = cluster_vid + cluster_aud + cluster_txt
        else:
            loss_val = 0.0
        # save features B x Pair x D
        # if epoch == 0:
        #     for param in model.parameters():
        #         print(param)
        # for key, param in model.state_dict().items():
        #     print(key,"\t",param.size())
        loss_cont = loss
        if args.kmeans == 1:
            loss+=loss_val*args.clu_lamb

        if args.recon:
            loss += recon_loss
    if apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    opt.step()
    scheduler.step()
    return loss.item(),queue_v, use_the_queue, centroid, loss_cont, loss_val, recon_loss

def Eval_retrieval(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for data in eval_dataloader:
            video = data['video'].cuda()
            audio = data['audio'].cuda()
            nframes = data['nframes'].cuda()
            text = data['text'].cuda()
            if args.tri_modal_fuse==1: # AVLnet-Text
                audio_text, video = model(video, audio, nframes, text)
                print("tri_modal_fuse=ON")
                m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
            else:
                if args.recon==1:
                    audio, video, text, recon_loss = model(video, audio, nframes, text)
                    print("tri_modal_fuse=ON recon=ON")
                else:
                    audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
                    print("tri_modal_fuse=ON recon=OFF")

                if args.eval_msrvtt==1:
                    audio_video=video+audio
                else:
                    audio_video = video+audio
                m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
            task = data['task'].cuda()
            metrics = compute_metrics(m, task, args.eval_lang_retrieval, args.eval_msrvtt)
            r1,r5,r10,rm = print_computed_metrics(metrics)
            return r1,r5,r10,rm

batch_time = AverageMeter()
youtube_load_time = AverageMeter()
data_time = AverageMeter()
queue_v = None
queue_a = None
queue_t = None

print("start")
file_path = os.path.join(args.checkpoint_dir, 'val_logs.csv')
writer_loss_whole=SummaryWriter(os.path.join(args.checkpoint_dir,"runs"))

with open(file_path,'a') as csv_file:
    fieldnames = ['epoch','R@1','R@5','R@10']
    writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
    if csv_file.tell() == 0:
        writer.writeheader()
    last_loss = 1000000.0
    flag_break=0
    for epoch in range(args.epochs):
        save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
            else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1

        if args.pretrain_path == '':
            writer_loss = SummaryWriter(os.path.join(args.checkpoint_dir, f"runs/epoch_{epoch}"))
        else:
            writer_loss = SummaryWriter(os.path.join(args.checkpoint_dir, f"runs/epoch_{save_epoch}"))
        if args.eval_youcook:
            r1,r5,r10,rm = Eval_retrieval(net, dataloader_val, 'YouCook2')
            writer.writerow({'epoch':save_epoch,'R@1':r1,'R@5':r5,'R@10':r10})
        if args.eval_msrvtt:
            r1,r5,r10,rm = Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
            writer.writerow({'epoch':save_epoch,'R@1':r1,'R@5':r5,'R@10':r10})
        if args.verbose:
            print('Epoch: %d' % epoch)
        end_time = time.time()
        print("middle")
        if args.withMLP==1:
            e_size = args.project_dim
        else:
            e_size = args.embd_dim
        queue_l = int(args.queue_size)*(int(args.n_pair/2))*args.batch_size
        if args.use_queue==1 and epoch >= args.start_queue and queue_v is None:  # will start at epoch 15
            queue_v = th.zeros(
                queue_l,
                e_size,
            ).cuda()

        save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
            else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
        use_the_queue = False
        centroid = None
        net.train() #追加

        @profile
        def learn(queue_v,use_the_queue,centroid):
            running_loss = 0.0
            for i_batch, sample_batch in tqdm(enumerate(dataloader),total=len(dataloader)):
                iteration = epoch * len(dataloader) + i_batch  # 0
                # end_time=time.time()

                batch_loss,queue_v,use_the_queue,centroid, loss_cont, loss_val, loss_recon = TrainOneBatch(net, optimizer, sample_batch, loss_op,queue_v,use_the_queue,scheduler, epoch,i_batch,centroid, apex)
                # process_batch_time = time.time() - end_time
                # batch_time.update(process_batch_time)
                running_loss += batch_loss
                writer_loss.add_scalar("train_it", loss_cont+loss_val+loss_recon, i_batch)
            
            # sys.exit()
            return batch_loss,queue_v,use_the_queue,centroid, loss_cont, loss_val, loss_recon, running_loss,i_batch
        batch_loss,queue_v,use_the_queue,centroid, loss_cont, loss_val, loss_recon,running_loss,i_batch = learn(queue_v,use_the_queue,centroid)
        
        if args.pretrain_path == '':
            writer_loss_whole.add_scalar("train_ep",running_loss/i_batch,epoch)
        else:
            writer_loss_whole.add_scalar("train_ep",running_loss/i_batch,save_epoch)
        save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
                    else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] *= args.lr_decay
        if args.checkpoint_dir != '':
            path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(save_epoch))
            net.module.save_checkpoint(path)
            if args.apex_level == 1:
                amp_checkpoint = {'net': net.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                'amp': amp.state_dict()}
                th.save(amp_checkpoint, os.path.join(args.checkpoint_dir, 'amp_checkpoint.pt'))
                
        if epoch > 30:
            if last_loss-(running_loss/i_batch) < 0.001:
                flag_break=flag_break+1
                last_loss = running_loss/i_batch
            else:
                last_loss = running_loss/i_batch
            if flag_break==3:
                break

# writer_loss.close()
if args.eval_youcook:
    Eval_retrieval(net, dataloader_val, 'YouCook2')
if args.eval_msrvtt:
    Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')

whole_time = time.time() - whole_time
print(whole_time)
