import torch
import torch.nn as nn
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(model_train, model, loss_fn, optimizer, epoch, epoch_step, epoch_step_val, gen,
 genval, Epoch, cuda,model_save):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, (data, target) in enumerate(gen):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()


            optimizer.zero_grad()
            outputs = model(*data)
            # print(outputs)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)

            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            # losses.append(loss.item())
            # total_loss += loss.item()
            loss.backward()
            optimizer.step()

            total_loss      += loss.item()
            # total_accuracy  += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                # 'acc'       : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:

        for iteration, (data, target) in enumerate(genval):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()


            optimizer.zero_grad()
            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)

            loss_val = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            # losses.append(loss.item())
            # total_loss += loss.item()
            # loss.backward()
            # optimizer.step()

            val_loss     += loss_val.item()
            # total_accuracy  += accuracy.item()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1), 
                                # 'acc'       : total_accuracy / (iteration + 1),
                                # 'lr'        : get_lr(optimizer)
                                })
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), model_save + '/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), 
    total_loss / epoch_step, val_loss / epoch_step_val))








    #     for iteration, batch in enumerate(genval):
    #         if iteration >= epoch_step_val:
    #             break
    #         images_val, targets_val = batch[0], batch[1]

    #         with torch.no_grad():
    #             if cuda:
    #                 images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
    #                 targets_val = torch.from_numpy(targets_val).type(torch.FloatTensor).cuda()
    #             else:
    #                 images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
    #                 targets_val = torch.from_numpy(targets_val).type(torch.FloatTensor)
    #             optimizer.zero_grad()
    #             outputs = nn.Sigmoid()(model_train(images_val))
    #             output  = loss(outputs, targets_val)

    #             equal       = torch.eq(torch.round(outputs), targets_val)
    #             accuracy    = torch.mean(equal.float())

    #         val_loss            += output.item()
    #         val_total_accuracy  += accuracy.item()

    #         pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
    #                             'acc'       : val_total_accuracy / (iteration + 1)})
    #         pbar.update(1)
            
    # print('Finish Validation')
    # print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    # print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    # torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
