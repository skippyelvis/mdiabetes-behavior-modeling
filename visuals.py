import matplotlib.pyplot as plt

def make_param_title(report):
    model = report["model_kw"]
    model_str = 

def plot_response_evaluations(report):
    res = report["results"]
    fig, ax = plt.subplots(nrows=len(res), edgecolor="w", facecolor="w")
    for i, r in enumerate(res):
        foo = ax[i].imshow(r, cmap="twilight", vmin=-3, vmax=3)
    ax[len(ax)//2].set_ylabel("Question #")
    ax[-1].set_xlabel("Response Week #")
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(foo, cax=cbar_ax)
    fig.suptitle("Error in Predicted Responses per User by Week")
    return (fig, ax)