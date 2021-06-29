import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ATE.transformations import logmap_so3


def viz_pose_grads_sep(
    cam_ids,
    gt,
    poses,
    rotGT,
    rotPose,
    t_grad,
    R_grad,
    bounds,
    arrow_length,
    global_step,
) -> Figure:
    num_poses = len(cam_ids)
    n_rows, n_cols = num_poses, 4
    figsize = np.array(plt.figaspect(n_rows / n_cols)) * 4
    fig = plt.figure(
        figsize=figsize,
    )  #
    fig.suptitle(f"Pose Gradients Step: {global_step}")
    marker_size = 200
    # Translation
    # cm = plt.get_cmap("tab20")
    # colors = cycler('color', 'bgrcmyk')
    # colors = cycle()
    # colors = cycle('bgrcmyk')
    # colors = iter(cm(np.linspace(0, 1, 6)))
    gt_color, pred_color, grad_color = "#1f77b4", "#ff7f0e", "#2ca02c"
    for i, cam_id in enumerate(cam_ids):
        # c = next(colors)

        ax1 = fig.add_subplot(n_rows, n_cols, i * n_rows + 1, projection="3d")
        ax2 = fig.add_subplot(
            n_rows,
            n_cols,
            i * n_rows + 2,
        )

        # ax4 = fig.add_subplot(224)
        ax1.axes.set_xlim3d(left=-bounds, right=bounds)
        ax1.axes.set_ylim3d(bottom=-bounds, top=bounds)
        # ax1.axes.set_zlim3d(bottom=-bounds, top=bounds)

        # 3D
        # GT pose without rotation
        ax1.scatter(gt[i, 0, 3], gt[i, 1, 3], gt[i, 2, 3], label=f"GT {cam_id:02}", color=gt_color, s=marker_size)
        # ax1.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
        # Pred translation with translation gradient
        # ax1.text2D(0.05, 0.95, "2D Text", transform=ax1.transAxes)
        ax1.text2D(
            0.5,
            -0.1,
            f"GT: {gt[i, :3, 3]}\nPred:{poses[i, :3, 3]}\nGrad: {t_grad[i]}",
            size=12,
            ha="center",
            transform=ax1.transAxes,
        )
        ax1.scatter(
            poses[i, 0, 3], poses[i, 1, 3], poses[i, 2, 3], label=f"Pred {cam_id:02}", color=pred_color, s=marker_size
        )
        ax1.quiver(
            poses[i, 0, 3],
            poses[i, 1, 3],
            poses[i, 2, 3],
            t_grad[i, 0],
            t_grad[i, 1],
            t_grad[i, 2],
            colors=grad_color,
            length=arrow_length,
            normalize=True,
            label=f"grad {cam_id:02}",
            linewidth=3,
        )
        # 2D
        # GT pose without rotation
        ax2.scatter(gt[i, 0, 3], gt[i, 1, 3], color=gt_color, label=f"GT {cam_id:02}", s=marker_size)
        ax2.scatter(poses[i, 0, 3], poses[i, 1, 3], color=pred_color, label=f"Pred {cam_id:02}", s=marker_size)
        # Pred translation with translation gradient
        ax2.quiver(
            poses[i, 0, 3],
            poses[i, 1, 3],
            t_grad[i, 0],
            t_grad[i, 1],
            color=grad_color,
            label=f"grad {cam_id:02}",
            angles="xy",
        )
        ax1.set_title(f"3D Translation+Gradients Cam {i}")
        ax1.legend()
        ax2.set_title(f"XY Translation+Gradients Cam {i}")
        ax2.legend()

    # Rotation vectors in axis angle
    for i, cam_id in enumerate(cam_ids):
        ax3 = fig.add_subplot(n_rows, n_cols, i * n_rows + 3, projection="3d")
        ax4 = fig.add_subplot(n_rows, n_cols, i * n_rows + 4)
        rotGT_so3 = logmap_so3(rotGT[i])
        rotPose_so3 = logmap_so3(rotPose[i])
        rotGrad_so3 = R_grad[i]

        # max_length = max(np.linalg.norm(rotGT_so3), np.linalg.norm(rotPose_so3), np.linalg.norm(rotGrad_so3))
        # ax3_bounds = 2 * max_length
        # ax3.axes.set_xlim3d(left=-ax3_bounds, right=ax3_bounds)
        # ax3.axes.set_ylim3d(bottom=-ax3_bounds, top=ax3_bounds)
        # ax3.axes.set_zlim3d(bottom=-ax3_bounds, top=ax3_bounds)

        ax3.text2D(
            0.5,
            -0.1,
            f"GT: {rotGT_so3}\nPred: {rotPose_so3}\nGrad: {rotGrad_so3}",
            size=12,
            ha="center",
            transform=ax3.transAxes,
        )

        # GT
        ax3.scatter(rotGT_so3[0], rotGT_so3[1], rotGT_so3[2], color=gt_color, label=f"GT {cam_id:02}", s=marker_size)
        # Predicted pose
        ax3.scatter(
            rotPose_so3[0],
            rotPose_so3[1],
            rotPose_so3[2],
            # length=arrow_length,
            color=pred_color,
            label=f"pred {cam_id:02}",
            s=marker_size,
        )

        # Rotation gradient attached to predicted pose location
        arrow_length = np.linalg.norm(rotPose_so3 - rotGT_so3) * 0.5
        print(f"rotPose_so3 {rotPose_so3}")
        print(f"rotGT_so3 {rotGT_so3}")
        print(f"norm {arrow_length}")
        ax3.quiver(
            rotPose_so3[0],
            rotPose_so3[1],
            rotPose_so3[2],
            rotGrad_so3[0],
            rotGrad_so3[1],
            rotGrad_so3[2],
            length=arrow_length,
            normalize=True,
            linewidth=4,
            color=grad_color,
            label=f"grad {cam_id:02}",
        )

        ax3.set_title(f"3D Rotation+Gradients Cam {i}")
        ax3.legend()

        # 2D
        # GT pose without rotation
        ax4.scatter(rotGT_so3[0], rotGT_so3[1], color=gt_color, label=f"GT {cam_id:02}", s=marker_size)
        ax4.scatter(rotPose_so3[0], rotPose_so3[1], color=pred_color, label=f"Pred {cam_id:02}", s=marker_size)
        # Pred translation with translation gradient
        ax4.quiver(
            rotPose_so3[0],
            rotPose_so3[1],
            rotGrad_so3[0],
            rotGrad_so3[1],
            color=grad_color,
            label=f"grad {cam_id:02}",
            angles="xy",
        )
        ax4.set_title(f"XY Rotation+Gradients Cam {i}")
        ax4.legend()

    return fig
