import os



def save_image(plt, img_name, file_dir):
    
        """
        Helper function to save images of the obtained plots.
        """
        image_dir=file_dir

        os.makedirs(image_dir, exist_ok=True)
        plt.savefig(os.path.join(image_dir, img_name))
        print(f"Image saved to '{image_dir}'")

