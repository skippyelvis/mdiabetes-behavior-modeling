import matplotlib.pyplot as plt

def make_param_title(title, report):
    model = report["model_kw"]
    lfn = model["lossfn"]
    lr = model["opt_kw"]["lr"]
    hid = model["hidden_size"]
    model_str = f"Loss={lfn}, LR={lr}, Hidden={hid}"
    return f"{title}\n{model_str}"

def plot_training_curves(report):
    curves = res["loss"]
    
def subplots(nrows=1, ncols=1, **kw):
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, 
        edgecolor="w", facecolor="w", **kw)
    return fig, ax
                 
class Plotter:

    def response_evaluations(report):
        res = report["results"]
        fig, ax = subplots(nrows=len(res))
        for i, r in enumerate(res):
            foo = ax[i].imshow(r, cmap="twilight", vmin=-3, vmax=3)
        ax[len(ax)//2].set_ylabel("Question #")
        ax[-1].set_xlabel("Response Week #")
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(foo, cax=cbar_ax)
        title = make_param_title(
            "Error in Predicted Responses per User by Week",
            report
        )
        ax[0].set_title(title, y=1.05)
        return (fig, ax)
    
    def training_loss(report):
        loss = report["loss"]
        fig, ax = subplots()
        for i in range(loss.shape[1]):
            curve = loss[:,i]
            ax.plot(curve)
        title = make_param_title(
            "Error in Predicted Responses per User by Week",
            report
        )
        ax.set_title(title, y=1.05)
        ax.set_ylabel("Error")
        ax.set_xlabel("Training epoch")
        return (fig, ax)