import pickle
import pathlib
current_path = pathlib.Path(__file__).parent.absolute()

print('Models are being loaded from: %s'%current_path)

def get_model(MODEL_ARCH, NUM_CLASSES):
    if MODEL_ARCH == 'CustomCNN':
        from models.CustomCNN import CustomCNN
        return CustomCNN(num_classes = NUM_CLASSES)
    elif MODEL_ARCH == 'ResNet18':
        from models.ResNet import resnet18
        return resnet18(num_classes = NUM_CLASSES, pretrained = True)
    elif MODEL_ARCH == 'ResNet34':
        from models.ResNet import resnet34
        return resnet34(num_classes = NUM_CLASSES, pretrained = True) 
    #### Add more code to add functionality to load other models #####
