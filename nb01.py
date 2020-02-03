from fastai.vision import *

# fastAI lesson 1, build your own classifier for your own categories based
# on resnet34 and resnet50

path = Path('.')
path_img = path/'images'
classes = ['moth', 'butterfly']
for c in classes:
    print(c)
    verify_images(path_img/c, delete=True, max_size=500)

np.random.seed(42)


data = ImageDataBunch.from_folder(path_img, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4)
data.normalize(imagenet_stats)
print(data.classes)


# training the model using resnet34

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')

interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(9, figsize=(15,11))

print(interp.most_confused(min_val=2))
