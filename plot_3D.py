"""
    3D plotting funtions
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns


def plot_3d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot several 3D surfaces next to each other."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('Sanity Check')
    print('------------------------------------------------------------------')
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Z shape:", Z.shape)
    print("X min/max:", np.min(X), np.max(X))
    print("Y min/max:", np.min(Y), np.max(Y))
    print("Z min/max:", np.min(Z), np.max(Z))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')


    #fig = plt.figure()
    #ax = Axes3D(fig)
    #X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    #Z = np.zeros_like(X)
    #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #fig.savefig('cifar10/trained_nets/TEST_PLOT_FAKE_DATA.pdf', dpi=300,
    #            bbox_inches='tight', format='pdf')

    # Dummy data
    #X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    #Z = np.zeros_like(X)  # Flat surface
    # Set up figure and 3D axes properly
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # <-- KEY difference
    # Aspect ratio (optional but recommended)
    #ax.set_box_aspect([1, 1, 0.5])
    # Plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='none', antialiased=True)
    # Colorbar
    fig.colorbar(surf, shrink=0.5, aspect=10)
    # Nice view
    ax.view_init(elev=30, azim=135)
    # Labels
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    f.close()
    if show: plt.show()

def plot_multiple(surf_files, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot multiple 2D contours and 3D surfaces next to each other with one colorbar each."""
    # Read file paths
    with open("./"+surf_files, 'r') as f:
        surf_file_list = [line.strip() for line in f if line.strip()]
    
    num_files = len(surf_file_list)
    print(f"Found {num_files} surface files.")

    data_list = []
    for surf_file in surf_file_list:
        with h5py.File(surf_file, 'r', libver='latest', swmr=True) as f:
        #with h5py.File(surf_file, 'r') as f:
            x = np.array(f['xcoordinates'][:])
            y = np.array(f['ycoordinates'][:])
            X, Y = np.meshgrid(x, y)

            if surf_name in f.keys():
                Z = np.array(f[surf_name][:])
            elif surf_name in ['train_err', 'test_err']:
                Z = 100 - np.array(f[surf_name][:])
            else:
                raise ValueError(f"{surf_name} not found in {surf_file}")

            data_list.append((surf_file, X, Y, Z))

    # ------------------------
    # 2D Contour plots
    # ------------------------
    fig2d, axes2d = plt.subplots(1, num_files, figsize=(5*num_files, 5))
    if num_files == 1:
        axes2d = [axes2d]  # Make iterable if only one file

    cf_list = []
    idx = 0
    print(len(args.labels))
    for ax, (surf_file, X, Y, Z) in zip(axes2d, data_list):
        cf = ax.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
        #ax.set_title(surf_file.split('/')[-1], fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cf_list.append(cf)
        if args.labels is not None and idx <= len(args.labels):
            x_pos = 0.29 + idx * (0.98 / num_files)
            fig2d.text(x=x_pos, y=-0.02, s=args.labels[idx], ha='center', fontsize=14)#, fontweight='bold')
        idx += 1

    ## After plotting your subplots
    fig2d.subplots_adjust(right=0.8, bottom=0.1)  # Make room for colorbar
    cbar_ax = fig2d.add_axes([0.99, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig2d.colorbar(cf_list[0], cax=cbar_ax)


    fig2d.tight_layout()
    contourf_pdf = str(surf_files)[:-4] + '_multiple_contours.pdf'
    fig2d.savefig(contourf_pdf, dpi=500, bbox_inches='tight', format='pdf')
    print(f"Saved 2D contour plots to {contourf_pdf}")

    # ------------------------
    # 3D Surface plots
    # ------------------------
    fig3d = plt.figure(figsize=(5*num_files, 5))
    axes3d = []
    for i in range(num_files):
        ax = fig3d.add_subplot(1, num_files, i+1, projection='3d')
        axes3d.append(ax)

    # surf_list = []
    # idx = 0
    # for ax, (surf_file, X, Y, Z) in zip(axes3d, data_list):
    #     surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', antialiased=True, vmin=vmin, vmax=vmax)
    #     #ax.set_title(surf_file.split('/')[-1], fontsize=10)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.view_init(elev=30, azim=135)
    #     surf_list.append(surf)
    #     if args.labels is not None and idx < len(args.labels):
    #         # For 3D plots, you must provide the z-coordinate explicitly
    #         x_pos = 0.3 + idx * (1 / num_files)
    #         fig3d.text(x=x_pos, y=-0.05, s=args.labels[idx], ha='center', fontsize=14, fontweight='bold')
    #     idx += 1


    # # Add ONE shared colorbar
    # fig3d.subplots_adjust(right=0.5, bottom=0.5)
    # cbar_ax = fig3d.add_axes([0.99, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # fig3d.colorbar(surf_list[0], cax=cbar_ax)
    # #cbar_ax.set_ylabel(surf_name)

    # fig3d.tight_layout()
    # surface_pdf = str(surf_files)[:-4] + '_multiple_3d_surfaces.pdf'
    # fig3d.savefig(surface_pdf, dpi=300, bbox_inches='tight', format='pdf')
    # print(f"Saved 3D surface plots to {surface_pdf}")




    fig3d = plt.figure(figsize=(6 * num_files, 7), constrained_layout=False)
    axes3d = [fig3d.add_subplot(1, num_files, i+1, projection='3d') for i in range(num_files)]

    surf_list = []
    for i, (ax, (surf_file, X, Y, Z)) in enumerate(zip(axes3d, data_list)):
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none',
                            antialiased=True, vmin=vmin, vmax=vmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=135)
        if args.labels is not None and i < len(args.labels):
            ax.set_title(args.labels[i], fontsize=12, pad=1, fontweight='bold')
        surf_list.append(surf)

    # Manually adjust spacing between subplots
    fig3d.subplots_adjust(wspace=0.25)  # Increase the wspace to reduce overlap

    # Shared colorbar to the right of all subplots
    fig3d.colorbar(surf_list[0], ax=axes3d, location='right', shrink=0.5, pad=0.1)

    surface_pdf = str(surf_files)[:-4] + '_multiple_3d_surfaces.pdf'
    fig3d.savefig(surface_pdf, dpi=500, bbox_inches='tight', format='pdf')


    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D loss surface')
    parser.add_argument('--surf_files', '-f', default='', help='A txt file that contains the paths to the h5 files that contains surface values')
    parser.add_argument('--dir_file', default='', help='The h5 file that contains directions')
    parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--zlim', default=10, type=float, help='Maximum loss value to show')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--labels', nargs='*', default=None, help='List of labels (one per plot) to write below each subplot')

    args = parser.parse_args()

    plot_multiple(args.surf_files, args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
