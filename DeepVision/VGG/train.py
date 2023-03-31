import tensorflow as tf
import argparse
from DeepVision.VGG.model import VGG
from DeepVision.utils.data import Datasets
from DeepVision.utils.helper import *


def vgg_preprocess(x):

    """

    Add padding or cropping to the input tensor to make it of size output_shape (which is input shape for the LeNet models)

    preprocess the input tensor

    input_shape: it is the shape of the input of the model.

    """


    x = tf.keras.layers.experimental.preprocessing.Resizing(
        224, 224
    )(x)

    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(x)

    return x


# make an instance of the LeNet class
def load_model(args):
    vgg_model = VGG(input_shape=args.input_shape, output_shape=args.output_shape)
    if args.model == "VGG_A":
        model = vgg_model.vgg_a()
    elif args.model == "VGG_A_LRN":
        model = vgg_model.vgg_a_lrn()
    elif args.model == "VGG_B":
        model = vgg_model.vgg_b()
    elif args.model == "VGG_C":
        model = vgg_model.vgg_c()
    elif args.model == "VGG_D":
        model = vgg_model.vgg_d()
    elif args.model == "VGG_E":
        model = vgg_model.vgg_e()
    return model


def main(args):
    model = load_model(args)

    # model_summary_only only
    if args.summary_only:
        model.summary()

    # if args.architecture:
    #     tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True)


    if args.train:

        dataset = Datasets().load_dataset(args.dataset)

        (x_train, y_train), (x_test, y_test) = dataset

        if args.preprocessing:
            x_train = vgg_preprocess(x_train)
            x_test = vgg_preprocess(x_test)

        model.compile(
            optimizer=optimizers(args.optimizer, args.learning_rate),
            loss=losses(args.loss),
            metrics=[metrics(args.metrics)],
        )

        model.fit(
            x_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_test, y_test),
        )



def arg_parse():
    args = argparse.ArgumentParser(add_help=True)

    args.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args.add_argument("--batch_size", type=int, default=256, help="Batch size")

    args.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    args.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use for training"
    )

    args.add_argument(
        "--model", type=str, default="AlexNet", help="Model to use for training"
    )

    args.add_argument(
        "--preprocessing",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Whether to use preprocessing or not",
    )

    args.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use for training"
    )

    args.add_argument(
        "--loss",
        type=str,
        default="categorical_crossentropy",
        help="Loss function to use for training",
    )

    args.add_argument(
        "--metrics", type=str, default="accuracy", help="Metrics to use for training"
    )

    args.add_argument(
        "--summary_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args.add_argument(
        "--architecture",
        type=bool,
        default=True,
        help="Whether to print model architecture or not",
    )

    args.add_argument(
        "--input_shape",
        type=tuple,
        default=(224, 224, 3),
        help="Input shape of the model",
    )

    args.add_argument(
        "--output_shape",
        type=int,
        default=10,
        help="Output shape of the model",
    )

    args = args.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    main(args)
