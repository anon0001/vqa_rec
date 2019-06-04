
from unet import FeatureLoss
from torchvision import transforms
from fastai.vision import *
from fastai.callbacks import *
import plot
from PIL import Image, ImageDraw


def NormalizeInverse(input):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)

        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        _transforms = []
        _transforms.append(transforms.Normalize(mean=mean_inv,
                             std=std_inv))
        _transforms.append(transforms.ToPILImage())

        t = transforms.Compose(_transforms)

        return t(input)


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(device=logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores




class LossModel(Callback):
    def on_epoch_begin(self, **kwargs):
        self.loss_model, self.total = 0, 0

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.loss_model / self.total)

    def __call__(self, item):
        self.loss_model += item
        self.total += 1

class LossReconstruction(Callback):
    def on_epoch_begin(self, **kwargs):
        self.loss_reconstruction, self.total = 0, 0

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.loss_reconstruction / self.total)

    def __call__(self, item):
        self.loss_reconstruction += item
        self.total += 1



class GradientClipping(LearnerCallback):
    def __init__(self,learn, clip):
        super(GradientClipping, self).__init__(learn)
        self.clip=clip

    def on_backward_end(self, **kwargs):
        nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)

class Precision(Callback):

    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        logits, _ = last_output
        v, a = last_target

        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*a.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * a)

        self.correct += scores.sum()
        self.total += a.size(0)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.correct / self.total)

class TotalLoss(nn.Module):
    def __init__(self, gamma_r, loss_m, loss_r, use_feat_loss):
        super().__init__()
        self.gamma_r = gamma_r
        self.loss_m = loss_m
        self.loss_r = loss_r

        self.use_feat_loss = use_feat_loss
        if self.use_feat_loss:
            self.rec_loss = FeatureLoss()
        else:
            self.rec_loss = F.mse_loss

    def forward(self, input, v, a):
        logits, g = input

        loss_model = instance_bce_with_logits(logits, a)
        loss_rec   = self.rec_loss(g, v)

        self.loss_m(loss_model.data)
        self.loss_r(loss_rec.data)
        return loss_model + self.gamma_r * loss_rec

class SaveModel(LearnerCallback):
    def __init__(self, learn, output):
        super(SaveModel, self).__init__(learn)
        self.best_accuracy_score = 0.0
        self.output = output
        self.model = learn

    def on_epoch_end(self, last_metrics, **kwargs):
        #last_metrics = val_loss, metric1, metric2,...
        accuracy_score = last_metrics[1]
        if accuracy_score > self.best_accuracy_score:
            old = os.path.join(self.output, 'model_%.4f.pth' % float(self.best_accuracy_score))
            if os.path.exists(old):
                os.remove(old)
            ret = self.model.save('model_%.4f' % float(accuracy_score), return_path=True)
            self.best_accuracy_score = accuracy_score

    def on_train_end(self,**kwargs):
        for p in ["plot_lr","plot_losses","plot_metrics"]:
            f = getattr(plot,p)
            fig, ax = f(self.learn.recorder)
            out = os.path.join(self.output,p)
            pickle.dump(ax, open(out, 'wb+'))
            fig.savefig(out)

        with open("all_scores.txt", "a+") as f:
            f.write('%s-%.4f \n'%(self.output,self.best_accuracy_score))

def train(model, train_loader, eval_loader, num_epochs, output, reconstruction, lr, gamma_r, layer, size,
          early_stop,
          finetune,
          use_feat_loss,
          dropout_hid,
          dropout_unet,
          use_one_cycle,
          ckpt,
          logger):

    logger.write("reconstruction %s, "
                 "gamma_r %s, "
                 "layer %s, "
                 "size %s, "
                 "use_feat_loss %s, "
                 "finetune %s, "
                 "dropout_hid %s, "
                 "dropout_unet %s, "
                 "use_one_cycle %s, "
                 % (reconstruction, str(gamma_r), str(layer), str(size), use_feat_loss, finetune, dropout_hid,
                    dropout_unet,
                    str(use_one_cycle),
                    ))



    lmod = LossModel()
    lrec = LossReconstruction()
    total_loss = TotalLoss(gamma_r, lmod, lrec, use_feat_loss)
    learn = Learner(DataBunch(train_dl=train_loader,
                              valid_dl=eval_loader),
                    model=model,
                    wd=0.0,
                    metrics=[Precision(), lmod, lrec],
                    loss_func=total_loss,
                    model_dir=output)


    if ckpt is not None:
        logger.write("Loading %s"% str(ckpt))
        learn.load(ckpt)

    if lr is None:
        learn.lr_find()
        _ = plot.plot(learn.recorder, suggestion=True)
        lr = learn.recorder.min_grad_lr

    gp = GradientClipping(learn,0.25)
    sm = SaveModel(learn, output=output)
    if use_one_cycle:
        fit = getattr(learn,"fit_one_cycle")
    else:
        fit = getattr(learn,"fit")
    # es = EarlyStoppingCallback(learn=learn, monitor='precision', min_delta=0.0001, patience=early_stop)
    print("lr",lr)
    fit(num_epochs, lr, callbacks=[gp,sm])


def evaluate(model, dataloader, eval_dset, gamma_r, use_feat_loss, output, ckpt, compute_acc = False):
    # filename = os.path.join(eval_dset.dataroot, eval_dset.coco_folder,
    #                         'COCO_%s%s_%s.jpg' % (eval_dset.name, str(eval_dset.year), str(73).zfill(12)))


    lmod = LossModel()
    lrec = LossReconstruction()
    total_loss = TotalLoss(None, lmod, lrec, 0)
    learn = Learner(DataBunch(train_dl=dataloader,
                              valid_dl=dataloader),
                    model=model,
                    wd=0.0,
                    metrics=[Precision(), lmod, lrec],
                    loss_func=total_loss,
                    model_dir=output,
                    )
    print("Loading ckpt", ckpt)
    learn.load(ckpt)

    valid = False
    if valid:
        print(learn.validate())
    else:
        correct = 0
        label2ans = dataloader.dataset.label2ans
        dictionary = dataloader.dataset.dictionary
        #get a random question
        index = np.random.randint(len(eval_dset))
        batch = eval_dset.__getitem__(index)

        #print question and GT
        q = batch[0][3]
        img = batch[1][0]
        gt = batch[1][1]
        gt = torch.max(gt, 0)[1].data
        print("Question :", " ".join(dictionary.detokenize(q[q != dictionary.padding_idx])))
        print("GT : ", label2ans[gt])

        #expanding dims as we only make a forward of one question
        for r in batch[0][1].keys():
            batch[0][1][r] = torch.from_numpy(batch[0][1][r]).unsqueeze(0)
        batch[0][3] = torch.from_numpy(batch[0][3]).unsqueeze(0)

        #run
        logits, g = learn.pred_batch(batch=batch)
        prediction = torch.max(logits, 1)[1].data
        print("Model answer :", label2ans[prediction[0]])
        print("Saving original image under name : ", str(index)+"_orig.jpg")
        print("Saving reconstructed image under name : ", str(index)+".jpg")
        im = NormalizeInverse(g[0].cpu().detach().data)
        im.save(str(index)+".jpg", "JPEG")
        im = NormalizeInverse(img.cpu().data)
        im.save(str(index)+"_orig.jpg", "JPEG")

        sys.exit()

        #
        # print("Processing ",str(len(dataloader.dataset)), "questions")

        # json_string = '['
        # for step, batch in enumerate(dataloader):
        #
        #     # ##### single
        #     batch = eval_dset.__getitem__(0)
        #     image_orig = batch[1][0]
        #     image_black = batch[0][2]
        #     f = transforms.ToPILImage()
        #     NormalizeInverse(image_orig).save("image_orig.jpg")
        #     NormalizeInverse(image_black).save("image_black.jpg")
        #
        #     inp = torch.unsqueeze(image_black, 0).cuda()  # 1,dim,x,x
        #     _, pre_pools = eval_dset.model.encode(inp)
        #     for r in pre_pools.keys():
        #         pre_pools[r] = pre_pools[r]
        #
        #
        #     # for r in batch[0][1].keys():
        #     #     batch[0][1][r] = batch[0][1][r].unsqueeze(0)
        #     batch[0][3] = batch[0][3].unsqueeze(0)
        #     batch[0][1] = pre_pools
        #
        #     ########
        #
        #     #input from dataset
        #     input, GT = batch
        #     question_id, _, _, _ = input
        #     # _, answer = GT
        #     _, answer, question_raw, index = GT
        #
        #
        #
        #     #run
        #     logits, g = learn.pred_batch(batch=batch)
        #     bs = logits.size(0)
        #     prediction = torch.max(logits, 1)[1].data
        #     if bs ==1:
        #         prediction = [prediction]
        #         question_raw = [question_raw]
        #         question_id = [question_id]
        #         index = [index]
        #     #process result
        #     save_visual = []
        #     for i in range(bs):
        #         json_string += '{"question_id":'+str(question_id[i])+'"answer":"'+label2ans[prediction[i]]+'"}, '
        #         save_visual.append([question_raw[i], NormalizeInverse(g[i].cpu().data),label2ans[prediction[i]],
        #                             question_id[i],
        #                             index[i]])
        #         NormalizeInverse(g[i].cpu().data).save("image_black_rec.jpg")
        #         print(label2ans[prediction[i]])
        #         print(question_raw[i])
        #     pickle.dump(save_visual,open("save_visual_single","wb+"))
        #     sys.exit()
        #     if compute_acc:
        #         correct += compute_score_with_logits(logits, answer).sum()
        # #sumbission file for (https://competitions.codalab.org/competitions/6961#participate-submit_results)
        # json_string = json_string[:-2] + ']'
        # with open("submission.json","w+") as f:
        #     f.write(json_string)
        # #accuracy
        # if compute_acc:
        #     print(correct.to(torch.float)/len(dataloader.dataset))

