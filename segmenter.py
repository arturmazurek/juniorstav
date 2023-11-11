import argparse
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from matplotlib.artist import Artist 
import matplotlib.pyplot as plt
import os
import cv2
import smallestenclosingcircle as sec

class SegmentationVisualization:
    """ Wrapper for a SAM visualization with matplotplib

    This is created as a small class to act as a data wrapper with a few methods.
    There's no additional value in it, as equally everything could've been using free functions.

    In behaviour it is similar the predictor example from SAM's predictor example or demo - that it 
    generates masks from position prompts. It aims to be as simplistic as possible while doing that
    to present that it's relatively simple to embed SAM in an interactive graph application.

    It calculates sphericity (roundness) of a mask found from a mouse click prompt and displays 
    the circle and the mask of that mask. It keeps a running list of inputs, so that you can add/remove
    areas in a last in - last out fashion.
    """

    def __init__(self, image_path, sam):
        """
        Parameters:
            image_path - disk path to an image to be analysed
            sam - an initialized Segment Anything Model instance
        """

        # first read the image, create the predictor and draw the clean image
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.sam = sam
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.image)

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)

        # we want the graph to react to mouse clicks, so connect this event
        self.fig.canvas.mpl_connect("button_release_event", self.on_click)

        # mouse click locations and the plot that visualizes them
        self.xpoints = []
        self.ypoints = []
        self.scatter_plot = None

        # last mask output from SAM and the plot used to display it
        self.current_mask = None
        self.mask_plot = None

        # circles and arrows with comments shown on the chart
        self.circle_shapes = []
        self.annotations = []

        plt.show()     

    def draw_mask(self):
        """Takes the last click positions, makes a SAM prompt of it and renders the output from SAM"""

        if self.mask_plot:
            # Remove the previous SAM visualization, we want to have only the last one visible
            self.mask_plot.remove()
            self.mask_plot = None

        if len(self.xpoints) == 0:
            return

        input_point = np.array([[self.xpoints[-1], self.ypoints[-1]]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        masks_scores = zip(masks, scores)
        masks_scores = sorted(masks_scores, key=lambda mask_score: mask_score[1])
        self.current_mask = list(masks_scores)[-1][0]

        # Draw the mask in light bluish colour
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = self.current_mask.shape[-2:]
        mask_image = self.current_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # and store the mask plot, so that it can be removed if a new one is needed
        self.mask_plot = self.ax.imshow(mask_image)

    def draw_points(self):
        """Draws the plot of points where mouse was clicked"""
        if self.scatter_plot:
            # All the mouse click positions are drawn simultanously on the graph, so 
            # remove the previous chart.
            self.scatter_plot.remove()
            self.scatter_plot = None

        # And create the new one with all the mouse click points
        self.scatter_plot = self.ax.scatter(self.xpoints, self.ypoints, color="cyan")

    def on_click(self, event):
        """Simply handles mouse click"""

        # Button 1 - left mouse button - adds a new point
        # Button 3 - right mouse button - removes the last point
        if event.button == 1:
            self.add_point(event.xdata, event.ydata)
        elif event.button == 3:
            self.remove_last_point()

    def add_point(self, x, y):
        """
        Handles adding a new point to the list of mouse click locations
        
        Adding a new point means generating and drawing a new SAM mask and 
        drawing a matching cirlce along all the known mouse click locations.
        """
        self.xpoints.append(x)
        self.ypoints.append(y)

        self.draw_mask()
        self.draw_points()
        self.add_circle()

        # Called to ask Matplotlib to redraw itself
        self.fig.canvas.draw_idle()

    def remove_last_point(self):
        """
        Handles removing the newest point from the list of mouse clicks
        
        It means removing the mask, the newest point from points graph and 
        the newest circle and its annotation.
        """

        # Since all arrays are in sync when adding, here xpoints is used to check
        # if something is left to remove
        if len(self.xpoints) > 0:
            self.xpoints.pop()
            self.ypoints.pop()

            self.draw_mask()
            self.draw_points()

            self.remove_circle()

            # Called to ask Matplotlib to redraw itself
            self.fig.canvas.draw_idle()

    def add_circle(self):
        """
        Fits a circle to the SAM mask and draws the circle with annotation
        
        Using the helper smallestenclosingcircle module this function finds a good fitting
        circle and draws it on the graph along with the sphericity value as annotation
        """

        # The width and height of the image in pixels
        h, w = self.current_mask.shape[-2:]

        # As simplest way this will hold all the points of the SEM mask to use them as fit points for a circle. 
        self.all_points = []
        for x in range(0, w):
            for y in range(0, h):
                row = self.current_mask[y]
                if row[x]:
                    self.all_points.append((x, y))

        # Asks the helper library for the circle parameters
        # circle[0] is center x
        # circle[1] is center y
        # circle[2] is radius
        # It's worth noting that since values in self.all_points are in pixels then these will also be in pixels
        circle = sec.make_circle(self.all_points)

        # Calculate and log the sphericity.
        # Circle area is easy to calculate and the mask area is simply the number of pixels in it because
        # all the numbers are in pixels.
        circle_area = np.pi * circle[2] * circle[2]
        sand_pixels = len(self.all_points)
        sphericity = sand_pixels / circle_area
        print(f"Circle area: {circle_area:.2f}")
        print(f"Sand pixels: {sand_pixels}")
        print(f"Sphericity: {sphericity:.2f}")
        print()
        
        # Draw the actual fitted circle. Store the shape in a local list, so that it can be removed when neeed
        color = "black"
        circle_shape = plt.Circle((circle[0], circle[1]), circle[2], fill=False, color=color)
        self.ax.add_artist(circle_shape)
        self.circle_shapes.append(circle_shape)

        # And draw the annotation, also store for removal.
        xy = (circle[0] + circle[2], circle[1])
        xytext = (xy[0] + 100, xy[1] - 100)
        annotation = self.ax.annotate(f"Sphericity: {sphericity:.2f}", xy=xy, xytext=xytext, arrowprops=dict(facecolor=color, shrink=0.01, edgecolor=color), color=color, fontsize=12)      
        self.annotations.append(annotation)

    def remove_circle(self):
        """Removes the last circle shape and annotation from the graph"""
        self.circle_shapes[-1].remove()
        self.circle_shapes.pop()

        self.annotations[-1].remove()
        self.annotations.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image to open")
    args = parser.parse_args()

    if not args.image_path or not os.path.exists(args.image_path):
        print(f"Given file at {args.image_path} doesn't exist")
        parser.print_help()
        exit(1)

    sam_checkpoint_file = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint_file):
        print(f"Checkpoint file {sam_checkpoint_file} doesn't exist.")
        print("Download from https://github.com/facebookresearch/segment-anything and put into working directory.")
        exit(2)

    # Initialize the SAM model like in their examples
    sam = sam_model_registry["default"](checkpoint=sam_checkpoint_file)
    sam.to(device="cuda")

    vis = SegmentationVisualization(args.image_path, sam)
