
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.backends import cudnn

from datetime import datetime
from dataloader import *
from models import *
from gt_convert_utils import *
from tqdm import tqdm


cuda = torch.cuda.is_available()
EPOCHS = 3
LOG_FREQUENCY = 10
lr = 0.001


CLASSES = cityscapes_labelsWithTrainID

#transformations on images and target masks
transform = transforms.ToTensor()
target_transform = ClassConverter(CLASSES)

# this is a dict to return the original class ids from our training labels
inverse_class_converter = target_transform.get_inverter()

transcityscape = Cityscapes('/localdata/Datasets/cityscapes',
                            split='train',
                            mode='fine',
                            target_type='semantic',
                            transform=None,
                            target_transform=target_transform,
                            size=(1024, 512),
                            image_transform=transform)

train_dataloader = DataLoader(transcityscape, batch_size=3,
                             shuffle=True, num_workers=4, collate_fn=None)

val_cityscape = Cityscapes('/localdata/Datasets/cityscapes',
                            split='val',
                            mode='fine',
                            target_type='semantic',
                            transform=None,
                            target_transform=target_transform,
                            size=(1024, 512),
                            image_transform=transform)

val_dataloader = DataLoader(val_cityscape, batch_size=3,
                             shuffle=True, num_workers=4, collate_fn=None)


#initializing model network and use of GPU if found
model = FCN8(len(CLASSES) + 1)
if cuda:
    model.cuda()

#Loss function and optimizer
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#logging training
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
log_dir = 'runs/train/' + current_time

#batch_size
bs = 2

cudnn.benchmark = True

def train(train_dataloader, val_dataloader, model, loss_criterion, EPOCHS):

    print("===================================...==================================\n")
    print(str(datetime.now()).split('.')[0], "Starting training and validation...\n")
    print("================================Results...==============================\n")


    val_dataiter = iter(val_dataloader)
    train_iterator = iter(train_dataloader)
    model.train()
    pbar = tqdm(total=EPOCHS)
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        val_running_loss = 0.0

        # set a progress bar later with tqdm
        for i in range(0, len(train_dataloader)):
            # zero the parameter gradients
            optimizer.zero_grad()
            

            try:
                inputs, labels = next(train_iterator)
            except StopIteration:
                # now the dataset is empty -> new epoch
                train_iterator = iter(train_dataloader)
                inputs, labels = next(train_iterator)
                pbar.update(1)
            # get the inputs
            # if cuda:
                # print("Trying to push stuff do gpu")
            # inputs.cuda()
            # labels.cuda()

            
            
            # forward + backward + optimize
            outputs = model(inputs.cuda())

            # measure loss
            loss = loss_criterion(outputs, labels.long().cuda())

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # update progress bar status  and print statistics
            running_loss += loss.item()
            if i % LOG_FREQUENCY == 0:  # print every 100 mini-batches
                pbar.write('training [%d, %5d] batch_loss: %f' %
                          (epoch + 1, i + 1, running_loss / (i + 1)))


            #validation
            if (i + 1) % LOG_FREQUENCY == 0:
                torch.cuda.empty_cache()

                

                try:
                    val_images, val_targets = next(val_dataiter)
                except StopIteration: 
                    # now the dataset is empty -> new epoch
                    val_dataiter = iter(val_dataloader)
                    val_images, val_targets = next(val_dataiter)
                with torch.no_grad():
                    outputs = model(val_images.cuda())
                    loss = loss_criterion(outputs, val_targets.long().cuda()) 
                    val_running_loss += loss.item()
                


            # logging training/validation metrics and prediction images in Tensorboard
            if (i + 1) % 500 == 0:
                # runs are organized by training start time
                
                writer = SummaryWriter(log_dir)
                
                writer.add_scalar('Validation loss', val_running_loss / (500 / LOG_FREQUENCY), i)

                writer.add_scalar('Training loss', running_loss / 500, i)
                val_running_loss = 0.0
                running_loss = 0.0   

                # show performance on some images of the test set in tensorboard 
                # writer.add_figure('Performance on test set example',
                #         visualize(model, val_dataiter, inverse_class_converter, numplots=4),global_step = i)
                writer.close()

        # saving weights of model after n iterations
        if (i + 1) % 100 == 0: 
            torch.save(model.state_dict(), '/localdata/Grace/Codes/model_weights/' +  current_time + '_' + str(i) + ".pth")

    # save final model
    torch.save(model.state_dict(), '/localdata/Grace/Codes/model_weights/' +  current_time + "_final.pth")
    pbar.close()

def visualize(model, dataloader, converter, numplots = 4):

    #test mode
    model.eval()

    fig, axes = plt.subplots(numplots, 2, figsize=(20, 15))
    plt.tight_layout()

    axes[0, 0].set_title('input image')
    axes[0, 1].set_title('prediction')
    # axes[0, 2].set_title('ground truth')

    for idx in np.arange(numplots):
        try:
            images, targets = next(iter(dataloader))
        except StopIteration:
            print("Validation Dataset is empty, skipping Visualization")
            model.train()
            return fig

        with torch.no_grad():
            if cuda:
                images.cuda()
            output_pred = model(images)

        imgs = 255*images[idx].permute(1, 2, 0).cpu().numpy()
        masks = torch.argmax(output_pred[idx].float(), dim = 0).cpu().numpy()
        preds = visualizeMask(masks, converter, imgs.astype(np.uint8))    

    for ax, img, mask, pred in zip(axes, imgs, masks, preds):
        ax[0].imshow(img)
        ax[1].imshow(pred)
        # ax[2].imshow(mask)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # ax[2].set_xticks([])
        # ax[2].set_yticks([])

    model.train()
    return fig


if __name__ == "__main__":
    train(train_dataloader, val_dataloader, model, loss_criterion, EPOCHS)