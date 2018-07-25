import mnist as mn

images, labels = mn.load_images(10, True)
print(images.shape)

twos = mn.filter_dataset(2, images, labels)
print(twos.shape)

mn.save_image(twos[0], 'fig.png')


# if __name__ == '__main__':
#     m = MNIST()
#     m.load_images(10, True)
#     images = m.get_images_of_digit(2)
#     m.display_image(images[1])
