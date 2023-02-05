import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_dct(input, masked_freq, output):
    gs = gridspec.GridSpec(1, 3)
    plt.figure(figsize=(20,20))
    plt.subplot(gs[0,0])
    input = input.squeeze(0).permute(1,2,0)
    plt.imshow(input)
    plt.subplot(gs[0,1])
    masked_freq = masked_freq.squeeze(0).permute(1,2,0)
    plt.imshow(masked_freq)
    plt.subplot(gs[0,2])
    output = output.squeeze(0).permute(1,2,0)
    plt.imshow(output)

def plot_bbox(img, anns):
    # draw bounding boxes
    for ann in anns:
        x, y, w, h = ann['bbox']
        plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], linewidth=2, color='r')