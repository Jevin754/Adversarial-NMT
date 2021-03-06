import utils.tensor
import utils.rand

import argparse
import dill
import logging

import torch
from torch import cuda
from torch.autograd import Variable
from model import NMT, EncoderRNN, Attn, LuongAttnDecoderRNN
from discriminator import Discriminator

import random 
#import time

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Starter code for JHU CS468 Machine Translation Final Project.")
parser.add_argument("--data_file", required=True,
                    help="File prefix for training set.")
parser.add_argument("--src_lang", default="de",
                    help="Source Language. (default = de)")
parser.add_argument("--trg_lang", default="en",
                    help="Target Language. (default = en)")
parser.add_argument("--model_file", required=True,
                    help="Location to dump the models.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=SGD)")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.001)")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum when performing SGD. (default=0.9)")
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
# feel free to add more arguments as you need

def main(options):

  use_cuda = (len(options.gpuid) >= 1)
  # if options.gpuid:
  #   cuda.set_device(options.gpuid[0])

  src_train, src_dev, src_test, src_vocab = torch.load(open(options.data_file + "." + options.src_lang, 'rb'))
  trg_train, trg_dev, trg_test, trg_vocab = torch.load(open(options.data_file + "." + options.trg_lang, 'rb'))

  batched_train_src, batched_train_src_mask, sort_index = utils.tensor.advanced_batchize(src_train, options.batch_size, src_vocab.stoi["<blank>"])
  batched_train_trg, batched_train_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_train, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)
  batched_dev_src, batched_dev_src_mask, sort_index = utils.tensor.advanced_batchize(src_dev, options.batch_size, src_vocab.stoi["<blank>"])
  batched_dev_trg, batched_dev_trg_mask = utils.tensor.advanced_batchize_no_sort(trg_dev, options.batch_size, trg_vocab.stoi["<blank>"], sort_index)

  print "preprocessing batched data..."
  processed_src = list()
  processed_trg = list()
  processed_src_mask = list()
  processed_trg_mask = list()
  for batch_i in range(len(batched_train_src)):
    if batched_train_src[batch_i].size(0) <= 35 and batched_train_trg[batch_i].size(0) <= 35:
      processed_src.append(batched_train_src[batch_i])
      processed_trg.append(batched_train_trg[batch_i])
      processed_src_mask.append(batched_train_src_mask[batch_i])
      processed_trg_mask.append(batched_train_trg_mask[batch_i])


  batched_train_src = processed_src
  batched_train_trg = processed_trg
  batched_train_src_mask = processed_src_mask
  batched_train_trg_mask = processed_trg_mask

  processed_src = list()
  processed_trg = list()
  processed_src_mask = list()
  processed_trg_mask = list()
  for batch_i in range(len(batched_dev_src)):
    if batched_dev_src[batch_i].size(0) <= 35 and batched_dev_trg[batch_i].size(0) <= 35:
      processed_src.append(batched_dev_src[batch_i])
      processed_trg.append(batched_dev_trg[batch_i])
      processed_src_mask.append(batched_dev_src_mask[batch_i])
      processed_trg_mask.append(batched_dev_trg_mask[batch_i])

  batched_dev_src = processed_src
  batched_dev_trg = processed_trg
  batched_dev_src_mask = processed_src_mask
  batched_dev_trg_mask = processed_trg_mask

  del processed_src, processed_trg, processed_trg_mask, processed_src_mask

  trg_vocab_size = len(trg_vocab)
  src_vocab_size = len(src_vocab)
  word_emb_size = 50
  hidden_size = 1024

  nmt = NMT(src_vocab_size, trg_vocab_size, word_emb_size, hidden_size,
            src_vocab, trg_vocab, attn_model = "general", use_cuda = True)
  discriminator = Discriminator(src_vocab_size, trg_vocab_size, word_emb_size, 
            src_vocab, trg_vocab, use_cuda = True)

  if use_cuda > 0:
    #nmt = torch.nn.DataParallel(nmt,device_ids=options.gpuid).cuda()
    nmt.cuda()
    #discriminator = torch.nn.DataParallel(discriminator,device_ids=options.gpuid).cuda()
    discriminator.cuda()
  else:
    nmt.cpu()
    discriminator.cpu()

  criterion_g = torch.nn.NLLLoss().cuda()
  criterion = torch.nn.CrossEntropyLoss().cuda()

  # Configure optimization
  optimizer_g = eval("torch.optim." + options.optimizer)(nmt.parameters(), options.learning_rate)
  optimizer_d = eval("torch.optim." + options.optimizer)(discriminator.parameters(), options.learning_rate)
  
  # main training loop
  f1 = open("train_loss", "a")
  f2 = open("dev_loss", "a")
  last_dev_avg_loss = float("inf")
  for epoch_i in range(options.epochs):
    logging.info("At {0}-th epoch.".format(epoch_i))
    # srange generates a lazy sequence of shuffled range

    train_loss_g = 0.0
    train_loss_d = 0.0
    train_loss_g_nll = 0.0
    train_loss_g_ce = 0.0
    train_loss_nll_batch_num = 0
    train_loss_ce_batch_num = 0
    for i, batch_i in enumerate(utils.rand.srange(len(batched_train_src))):
      if i == 1500:
         break
      # if i==5:
      #   break
      train_src_batch = Variable(batched_train_src[batch_i])  # of size (src_seq_len, batch_size)
      train_trg_batch = Variable(batched_train_trg[batch_i])  # of size (src_seq_len, batch_size)
      train_src_mask = Variable(batched_train_src_mask[batch_i])
      train_trg_mask = Variable(batched_train_trg_mask[batch_i])
      if use_cuda:
        train_src_batch = train_src_batch.cuda()
        train_trg_batch = train_trg_batch.cuda()
        train_src_mask = train_src_mask.cuda()
        train_trg_mask = train_trg_mask.cuda()

      # train discriminator
      sys_out_batch = nmt(train_src_batch, train_trg_batch, True).detach()
      _,predict_batch = sys_out_batch.topk(1)
      del _
      predict_batch = predict_batch.squeeze(2)
      real_dis_label_out = discriminator(train_src_batch, train_trg_batch, True)
      fake_dis_label_out = discriminator(train_src_batch, predict_batch, True)
      optimizer_d.zero_grad()
      loss_d_real = criterion(real_dis_label_out, Variable(torch.ones(options.batch_size*len(options.gpuid)).long()).cuda())
      loss_d_real.backward()
      loss_d_fake = criterion(fake_dis_label_out, Variable(torch.zeros(options.batch_size*len(options.gpuid)).long()).cuda())
      #loss_d_fake.backward(retain_graph=True)
      loss_d_fake.backward()
      loss_d = loss_d_fake.data[0]+loss_d_real.data[0]
      del loss_d_fake, loss_d_real
      logging.debug("D loss at batch {0}: {1}".format(i, loss_d))
      f1.write("D train loss at batch {0}: {1}\n".format(i, loss_d))
      optimizer_d.step()

      if use_cuda > 0:
        sys_out_batch = sys_out_batch.cuda()
        train_trg_batch = train_trg_batch.cuda()
      else:
        sys_out_batch = sys_out_batch.cpu()
        train_trg_batch = train_trg_batch.cpu()

      # train nmt
      sys_out_batch = nmt(train_src_batch, train_trg_batch, True)
      _,predict_batch = sys_out_batch.topk(1)
      predict_batch = predict_batch.squeeze(2)
      fake_dis_label_out = discriminator(train_src_batch, predict_batch, True)
      if random.random()>0.5:
        train_trg_mask = train_trg_mask.view(-1)
        train_trg_batch = train_trg_batch.view(-1)
        train_trg_batch = train_trg_batch.masked_select(train_trg_mask)
        train_trg_mask = train_trg_mask.unsqueeze(1).expand(len(train_trg_mask), trg_vocab_size)
        sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
        sys_out_batch = sys_out_batch.masked_select(train_trg_mask).view(-1, trg_vocab_size)
        loss_g = criterion_g(sys_out_batch, train_trg_batch)
        train_loss_g_nll += loss_g
        train_loss_nll_batch_num += 1
        f1.write("G train NLL loss at batch {0}: {1}\n".format(i, loss_g.data[0]))
      else:
        loss_g = criterion(fake_dis_label_out, Variable(torch.ones(options.batch_size*len(options.gpuid)).long()).cuda())
        train_loss_g_ce += loss_g
        train_loss_ce_batch_num += 1
        f1.write("G train CE loss at batch {0}: {1}\n".format(i, loss_g.data[0]))

      logging.debug("G loss at batch {0}: {1}".format(i, loss_g.data[0]))
      
      

      optimizer_g.zero_grad()
      loss_g.backward()
      
      # # gradient clipping
      torch.nn.utils.clip_grad_norm(nmt.parameters(), 5.0)
      optimizer_g.step()


      train_loss_d += loss_d
    train_avg_loss_g_nll = train_loss_g_nll / train_loss_nll_batch_num
    train_avg_loss_g_ce = train_loss_g_ce / train_loss_ce_batch_num
    train_avg_loss_d = train_loss_d / len(train_src_batch)
    logging.info("G TRAIN Average NLL loss value per instance is {0} at the end of epoch {1}".format(train_avg_loss_g_nll, epoch_i))
    logging.info("G TRAIN Average CE loss value per instance is {0} at the end of epoch {1}".format(train_avg_loss_g_ce, epoch_i))
    logging.info("D TRAIN Average loss value per instance is {0} at the end of epoch {1}".format(train_avg_loss_d, epoch_i))
      


    # validation -- this is a crude esitmation because there might be some paddings at the end
    # dev_loss_g_nll = 0.0
    # dev_loss_g_ce = 0.0
    # dev_loss_d = 0.0

    # for batch_i in range(len(batched_dev_src)):
    #   dev_src_batch = Variable(batched_dev_src[batch_i], volatile=True)
    #   dev_trg_batch = Variable(batched_dev_trg[batch_i], volatile=True)
    #   dev_src_mask = Variable(batched_dev_src_mask[batch_i], volatile=True)
    #   dev_trg_mask = Variable(batched_dev_trg_mask[batch_i], volatile=True)
    #   if use_cuda:
    #     dev_src_batch = dev_src_batch.cuda()
    #     dev_trg_batch = dev_trg_batch.cuda()
    #     dev_src_mask = dev_src_mask.cuda()
    #     dev_trg_mask = dev_trg_mask.cuda()

    #   sys_out_batch = nmt(dev_src_batch, dev_trg_batch, False).detach()
    #   _,predict_batch = sys_out_batch.topk(1)
    #   predict_batch = predict_batch.squeeze(2)
    #   real_dis_label_out = discriminator(dev_src_batch, dev_trg_batch, True).detach()
    #   fake_dis_label_out = discriminator(dev_src_batch, predict_batch, True).detach()

    #   if use_cuda > 0:
    #     sys_out_batch = sys_out_batch.cuda()
    #     dev_trg_batch = dev_trg_batch.cuda()
    #   else:
    #     sys_out_batch = sys_out_batch.cpu()
    #     dev_trg_batch = dev_trg_batch.cpu()

    #   dev_trg_mask = dev_trg_mask.view(-1)
    #   dev_trg_batch = dev_trg_batch.view(-1)
    #   dev_trg_batch = dev_trg_batch.masked_select(dev_trg_mask)
    #   dev_trg_mask = dev_trg_mask.unsqueeze(1).expand(len(dev_trg_mask), trg_vocab_size)
    #   sys_out_batch = sys_out_batch.view(-1, trg_vocab_size)
    #   sys_out_batch = sys_out_batch.masked_select(dev_trg_mask).view(-1, trg_vocab_size)
    #   loss_g_nll = criterion_g(sys_out_batch, dev_trg_batch)
    #   loss_g_ce = criterion(fake_dis_label_out, Variable(torch.ones(options.batch_size*len(options.gpuid)).long(),volatile=True).cuda())
    #   loss_d = criterion(real_dis_label_out, Variable(torch.ones(options.batch_size*len(options.gpuid)).long(),volatile=True).cuda()) + criterion(fake_dis_label_out, Variable(torch.zeros(options.batch_size*len(options.gpuid)).long(),volatile=True).cuda())
    #   logging.debug("G dev NLL loss at batch {0}: {1}".format(batch_i, loss_g_nll.data[0]))
    #   logging.debug("G dev CE loss at batch {0}: {1}".format(batch_i, loss_g_ce.data[0]))
    #   f2.write("G dev NLL loss at batch {0}: {1}\n".format(batch_i, loss_g_nll.data[0]))
    #   f2.write("G dev CE loss at batch {0}: {1}\n".format(batch_i, loss_g_ce.data[0]))
    #   logging.debug("D dev loss at batch {0}: {1}".format(batch_i, loss_d.data[0]))
    #   f2.write("D dev loss at batch {0}: {1}\n".format(batch_i, loss_d.data[0]))
    #   dev_loss_g_nll += loss_g_nll
    #   dev_loss_g_ce += loss_g_ce
    #   dev_loss_d += loss_d
    # dev_avg_loss_g_nll = dev_loss_g_nll / len(batched_dev_src)
    # dev_avg_loss_g_ce = dev_loss_g_ce / len(batched_dev_src)
    # dev_avg_loss_d = dev_loss_d / len(batched_dev_src)
    # logging.info("G DEV Average NLL loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss_g_nll.cpu().data[0], epoch_i))
    # logging.info("G DEV Average CE loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss_g_ce.cpu().data[0], epoch_i))
    # logging.info("D DEV Average loss value per instance is {0} at the end of epoch {1}".format(dev_avg_loss_d.data[0], epoch_i))
    # # if (last_dev_avg_loss - dev_avg_loss).data[0] < options.estop:
    # #   logging.info("Early stopping triggered with threshold {0} (previous dev loss: {1}, current: {2})".format(epoch_i, last_dev_avg_loss.data[0], dev_avg_loss.data[0]))
    # #   break
  torch.save(nmt, open("nmt.nll_{0:.2f}.epoch_{1}".format(train_avg_loss_g_nll.cpu().data[0], epoch_i), 'wb'), pickle_module=dill)
  torch.save(discriminator, open("discriminator.nll_{0:.2f}.epoch_{1}".format(train_avg_loss_d.data[0], epoch_i), 'wb'), pickle_module=dill)
  f1.close()
  f2.close()



if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)