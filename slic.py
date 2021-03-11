import cv2
import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.future import graph

def slic_rag(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #1.jpg
    labels1 = segmentation.slic(image, compactness=15, n_segments=4500, start_label=0)
    #2.jpg
    #labels1 = segmentation.slic(image, compactness=20, n_segments=3000, start_label=0)
    
    out1 = color.label2rgb(labels1, image, kind='avg', bg_label=0)
    g = graph.rag_mean_color(image, labels1)
    labels2 = graph.cut_threshold(labels1, g,15)
    out2 = color.label2rgb(labels2, image, kind='avg', bg_label=0)
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                           figsize=(6, 8))
    out2=cv2.cvtColor(out2,cv2.COLOR_BGR2RGB)
    ax[0].imshow(out1)
    ax[1].imshow(out2)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    return out2
