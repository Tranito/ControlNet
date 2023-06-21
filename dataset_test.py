from Cityscapes_dataset import CityscapesDataset

from torch.utils.data import DataLoader

training_set = CityscapesDataset(set='train')
print(f'{len(training_set)} images')

item = training_set[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

validation_set = CityscapesDataset(set='val')
print(f'{len(validation_set)} images')

item = validation_set[222]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

val_dataloader = DataLoader(validation_set, num_workers=0, batch_size=30, shuffle=False)

# for i in range(10):
#     images=next(iter(val_dataloader))
#     print(images.keys())


