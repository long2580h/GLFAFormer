import os
import time

import timm.models.deit
import torch
from tensorboardX import SummaryWriter
from torch import nn

from models import get_model
from validate import validate
from data import create_dataloader
from pytorchtools import EarlyStopping
# from earlystop import EarlyStopping
from networks.trainer import Trainer
# from networks.testKD import Trainer
from options.train_options import TrainOptions
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def kd(teachermodel, device, data_loader, val_loader):
    # teachermodel.eval()
    studentModel = timm.models.deit.deit3_base_patch16_224()
    studentModel.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(studentModel.parameters(), lr=1e-4)

    epochs = 20

    for epoch in range(epochs):
        for i, data in enumerate(data_loader):
            student_output = studentModel(data)
            loss = criterion(student_output, teachermodel)
            print("Train loss: {}", format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            studentModel.eval()
            ap, r_acc, f_acc, acc = validate(studentModel.model, val_loader)
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

    studentModel.save_networks('student.pth')

if __name__ == '__main__':
    #--------------------------
    graphviz1 = GraphvizOutput()
    graphviz1.output_file = 'optimize_parameters.png'
    graphviz2 = GraphvizOutput()
    graphviz2.output_file = 'Trainer.png'
    graphviz3 = GraphvizOutput()
    graphviz3.output_file = 'get_model.png'
    #--------------------------

    opt = TrainOptions().parse()
    val_opt = get_val_opt()


    # ---------------------------------------------------
    with PyCallGraph(output=graphviz3):
        random_number3 = get_model(opt.arch)
    # ---------------------------------------------------
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)
    model = Trainer(opt)
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "test"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001,
                                   verbose=True)
    # 早停策略，防止过拟合，当性能不再改善时就提前停止训练
    # patience 表示模型在验证集上连续多少个迭代中没有性能提升时，就触发早停法。
    # delta 表示模型在验证集上的性能提升阈值。
    # verbose 表示是否打印早停法的详细信息。
    start_time = time.time()
    print("Length of data loader: %d" % (len(data_loader)))

    #---------------------------------------------------------------------------------------------------------------------------
    # teacher = torch.load('E:/Pycharm/UniversalFakeDetect-mailn/UniversalFakeDetect-main/pretrained_weights/fc_weights.pth')
    # kd(teacher, torch.device("cuda" if torch.cuda.is_availabe else "cpu"), data_loader, val_loader)
    #---------------------------------------------------------------------------------------------------------------------------

    for epoch in range(opt.niter):

        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            #---------------------------------------------------
            with PyCallGraph(output=graphviz1):
                random_number1 = model.optimize_parameters()
            #---------------------------------------------------

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print("Iter time: ", ((time.time() - start_time) / model.total_steps))

            if model.total_steps in [10, 30, 50, 100, 1000, 5000, 10000] and False:  # save models at these iters
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks('5-8addViTResnet.pth')
            model.save_networks('model_epoch_%s.pth' % epoch)

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
        # ---------------------------------------------------
        with PyCallGraph(output=graphviz2):
            random_number2 = model.train()
        # ---------------------------------------------------
