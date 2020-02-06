
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
EPOCHS = 150
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
        val_counter = 0
        train_counter = 0

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
            train_counter += 1
            if i % LOG_FREQUENCY == 0:  # print every 100 mini-batches
                pbar.write('training epoch :%d at iteration %5d  with batch_loss: %f' %
                          (epoch + 1, i + 1, running_loss / train_counter))


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
                    val_counter += 1
                


            # # logging training/validation metrics and prediction images in Tensorboard
            # if (i + 1) % 500 == 0:
            #     # runs are organized by training start time
        try:
            val_images, val_targets = next(val_dataiter)
        except StopIteration: 
            # now the dataset is empty -> new epoch
            val_dataiter = iter(val_dataloader)
            val_images, val_targets = next(val_dataiter)
        
        writer = SummaryWriter(log_dir)
        
        writer.add_scalar('Validation loss', val_running_loss / val_counter, epoch)

        writer.add_scalar('Training loss', running_loss / train_counter, epoch)
        val_running_loss = 0.0
        running_loss = 0.0   
        train_counter = 0
        val_counter = 0

        # show performance on some images of the test set in tensorboard 
        writer.add_figure('Performance on test set example',
                visualize(model, val_images, inverse_class_converter, val_targets),global_step = epoch)
        writer.close()

        # saving weights of model after n iterations
        if (i + 1) % 100 == 0: 
            torch.save(model.state_dict(), '/localdata/Grace/Codes/model_weights/' +  current_time + '_' + str(i) + ".pth")

    # save final model
    torch.save(model.state_dict(), '/localdata/Grace/Codes/model_weights/' +  current_time + "_final.pth")
    pbar.close()

def visualize(net, images, converter, targets):
    '''
    This function produces a matplotlib figure that shows the networks segmentation output 
    on images of the evaluation dataset
    '''
    num_plots = len(images)
    model.eval() # set network into test mode (turn off dropout)
    fig = plt.figure(figsize=(10,num_plots*5), dpi=240)
    out = []
    with torch.no_grad():
        if cuda:
            images = images.cuda()
        out = model(images)
    for idx in np.arange(num_plots): 
        ax = fig.add_subplot(2*num_plots, 2, 2*idx+1, xticks=[], yticks=[])
        npimg = 255*images[idx].permute(1, 2, 0).cpu().numpy()
        nptarget = targets[idx].cpu().numpy()
        nppreds = torch.argmax(out[idx].float(), dim = 0).cpu().numpy()
        if idx > num_plots // 2 - 1:
            outimage = visualizeMask(nppreds, converter)
        else:
            outimage = visualizeMask(nppreds, converter, npimg.astype(np.uint8))
        targetimage = visualizeMask(nptarget, converter)
        plt.imshow(outimage)
        ax = fig.add_subplot(2*num_plots, 2, 2*idx+2, xticks=[], yticks=[])
        plt.imshow(targetimage)
    plt.tight_layout(pad=0)
    model.train() # back into training mode
    return fig


if __name__ == "__main__":
    train(train_dataloader, val_dataloader, model, loss_criterion, EPOCHS)