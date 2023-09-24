import torch, cv2, time, random, mediapipe as mp, numpy as np, pickle

from torchvision.io import read_image
from torch import nn
from torch.utils.data import DataLoader

print('Would you like to record new pose data? (y/n)')

while True:
    recording = input()

    if recording == 'y':
        record = True
        print('\n')
        break
    elif recording == 'n':
        recording == False
        print('\n')
        break
    else:
        continue



if recording == 'y':
    print(f"Select mode\n\n'c' for custom\nYou will choose the amount of hand shapes that you'd like to classify.\nYou will then name and record those hand shapes.\n\n'd' for default\nYou will use the default 5 hand shapes of\n'c','palm','fist','pointer','pinkie'")


    poses = []
    num_gest = 0

    def set_gestures(): # function to allow user to customize their hand gestures
        while True:
            print('\nHow many hand shapes would you like to train for?')
            num_gest = input()
            try:
                num_gest = int(num_gest)
            except:
                print('\nPlease enter an integer for the number of hand shapes')
                continue

            for i in range(num_gest):
                print(f'\nPlease enter the name of your {i+1}/{num_gest} hand shape')
                gest = input()
                poses.append(gest)
                

            print(f'\nYou have chosen to train for {num_gest} hand shapes\nThe hand shapes are named\n{poses}')
            break
        return num_gest


    while True: # decide if you would like to create custom gestures or standard
        mode = input()

        if mode == 'd':
            poses = ['c','palm','fist','pointer','pinkie']
            num_gest = 5
            print(f'\nYou have chosen the default 5 hand shapes of {poses}')
            break
        elif mode == 'c':
            print(f"\nYou have chosen to record your own hand shapes")
            num_gest = set_gestures()
            break
        else:
            print("\nPlease enter 'c' for custom, or 'd' for default")

    with open('ModelWeights\\final_gest_count.pkl', 'wb') as f:
        pickle.dump([num_gest,poses], f)
elif recording == 'n':
    with open('ModelWeights\\final_gest_count.pkl','rb') as f:
        num_gest,poses = pickle.load(f)


mphands = mp.solutions.hands #Import the MediaPipe Hands model.
hands = mphands.Hands() #Initialize the MediaPipe Hands model, with default parameters.
mp_drawing = mp.solutions.drawing_utils #Import the drawing utilities from MediaPipe.
# Initialize the video capture object, using the default camera (index 0) on the system
cap = cv2.VideoCapture(0) 

# reads a frame from the video capture 
_, frame = cap.read()
#  retrieves the shape of the image (frame) in the format of (height, width, channels)
h, w, c = frame.shape




# Initialize variables
hand_image = np.zeros(frame.shape)
training_images = []

def set_camera(full,hand):
    #attempt to show the correct images
    try:
        cv2.imshow("camera", full)
        cv2.imshow("hand",hand)
        cv2.waitKey(1)

    # if hand not in the frame, show black frames in both windows
    except:
        cv2.imshow("camera", np.zeros(frame.shape))
        cv2.imshow("hand",np.zeros((256,256)))
        cv2.waitKey(1)

def find_hands(state,batch_size,pose_num,i):
    _, frame = cap.read() # Captures a frame from the video capture
    frame2 = np.copy(frame)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    hand_image = np.zeros((256,256,3))
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max, y_max, x_min, y_min = 0, 0, w, h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x_max, x)
                x_min = min(x_min, x)
                y_max = max(y_max, y)
                y_min = min(y_min, y)

            y_min, y_max, x_min, x_max = y_min-20, y_max+20, x_min-20, x_max+20

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

            if state == 'gather':
                cv2.putText(frame, f'{i} \ {batch_size*64} photos taken', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 1)

            # Crop and resize hand image
            try:
                hand_image = frame2[(y_min):(y_max), (x_min):(x_max)]
                hand_image = cv2.resize(hand_image,(256,256))
                hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

                torch_hand = np.copy(hand_image)
                torch_hand = torch.from_numpy(torch_hand)
                torch_hand = torch_hand.to(torch.float32)
                
                
                if state == 'gather':
                    torch_hand = torch_hand[None]
                    training_images.append((torch_hand,pose_num))
                
                elif state == 'display':
                    torch_hand = torch_hand.unsqueeze(0)
                    torch_hand = torch_hand.unsqueeze(0)

                    # get the predicted pose label from the model
                    outputs = model(torch_hand)
                    _, predicted = torch.max(outputs.data, 1)

                    # add the predicted pose label to the hand image
                    cv2.putText(hand_image, poses[int(predicted)], (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 1)
                
            except:
                hand_image = np.zeros((256,256,3))
            break
    return frame, hand_image
    
def get_images(pose_num,batch_size = 5):
    """
    Capture a specified number of hand images for a given pose and add them to training_images list.

    Args:
        pose_num (int): The index of the pose being captured.
        batch_size (int): The number of images to capture.

    Returns:
        None
    """
    #The 64 is the default batch size used in training the model.
    for i in range(64*batch_size):
        # get images
        frame, hand_image = find_hands('gather',batch_size,pose_num,i)
        # Display images            
        set_camera(frame,hand_image)
  
        

# define a function to delay the photos taken and show the hand landmarks
def delay_photos(count):
    #clean the camera to show next pose
    frame.fill(0)
    # display the pose label
    cv2.putText(frame, "show pose " +poses[count], (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 1)
    
    # show the camera and hand images in separate windows
    try:
        cv2.imshow("camera", frame)
        cv2.imshow("hand",hand_image)
        cv2.waitKey(1)
    except:
        cv2.imshow("camera", np.zeros(frame.shape))
        cv2.imshow("hand",np.zeros((256,256)))
        cv2.waitKey(1)
    # wait for 5 seconds
    time.sleep(5)


# create the camera and hand image windows
cv2.imshow("camera", np.zeros((256,256)))
cv2.imshow("hand",np.zeros((256,256))) 
cv2.setWindowProperty('hand', cv2.WND_PROP_TOPMOST, 1)
cv2.setWindowProperty('camera', cv2.WND_PROP_TOPMOST, 1)


if recording == 'y':
    # loop through all the gesture labels
    for count in range(num_gest):
        # delay the photos and show the hand landmarks
        delay_photos(count)
        # get images for training or testing
        get_images(count,2)



class AlexNet(nn.Module): #setup alexnet model
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),            
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        
        self.Conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.Conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.ReLU(),
        )
        
        self.FC1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 2048),
            nn.ReLU(),
        )
        
        self.FC2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
        )
        
        self.FC3= nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
            #nn.Softmax(dim=1),
            
        )
        
    def forward(self, x):
        out = self.Conv1(x)                       
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        out = out.reshape(out.size(0), -1)
        out = self.FC1(out)
        out = self.FC2(out)
        out = self.FC3(out)
        return out

##training data##
device = "cuda" if torch.cuda.is_available() else "cpu"#show device being used, ideally cuda
print(f"Using {device} device")

model = AlexNet(num_gest)
model.to(device)

epochs = 10 #set number of epochs
batch = 64 #set batch size

loss_fn = torch.nn.CrossEntropyLoss() #define the loss function
learning_rate = 0.002 #set learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)   #define the optimizer

def train_loop(dataloader, ml, loss_fn, opt):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (images, labels) in enumerate(dataloader): #for each batch of data

        #send images and labels to device
        images = images.to(device) 
        labels = labels.to(device)

        # Compute prediction and loss
        y_pred = ml(images)
        loss = loss_fn(y_pred, labels)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 100 == 0: 
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}")

        correct += (y_pred.argmax(1) == labels).type(torch.float).sum().item()
    return loss.item(), correct / size
            
def test_loop(dataloader, ml, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:  #for testing data
           
            #send images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Compute prediction 
            y_pred = ml(images)
            
            #calculate loss
            test_loss += loss_fn(y_pred, labels).item()
            correct += (y_pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct

def train_network(train, test,m,l_fn,opt,epochs):
    start_time = time.time()#set the start time for training
    train_L = []
    train_A = []

    test_L = []
    test_A = []
    for t in range(epochs): #for each epoch, run training and test loop
        print(f"Epoch {t+1}")
        train_res = train_loop(train, m, l_fn, opt)
        train_L.append(train_res[0])
        train_A.append(train_res[1])

        test_res = test_loop(test, m, l_fn)
        test_L.append(test_res[0])
        test_A.append(test_res[1])

    print("Done!")
    print("Time to train: " + str(time.time()-start_time))
    return train_L,test_L , train_A, test_A

if recording =='y':
    dataset = training_images

    random.shuffle(dataset) #shuffle data
    # Split the dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Define the data loader for the training set
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    # Define the data loader for the validation set
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Define the data loader for the test set
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    print(training_images[0][0].shape)
    all_res = []
    results = train_network(train_loader, test_loader, model,loss_fn,optimizer,epochs) #train and test the network
    all_res.append(results)

    torch.save(model.state_dict(), 'ModelWeights\\final_model_weights.pth')
    torch.save(num_gest,'ModelWeights\\final_gest_count.pth')



# define a function to display results
def display_results():
    #get images
    frame, hand_image = find_hands('display',64,1,1)
    #Display images            
    set_camera(frame,hand_image)
    
    

model = AlexNet(num_gest)
model.load_state_dict(torch.load('ModelWeights\\final_model_weights.pth'))
model.eval()

while True:
    display_results()
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Turning off camera.")
        cap.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break