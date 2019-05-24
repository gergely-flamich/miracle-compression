import tensorflow as tf
from sonnet import AbstractModule, Transposable, reuse_variables, \
    Conv2D, Conv2DTranspose
import tensorflow_probability as tfp
tfd = tfp.distributions

DATA_FORMAT_NHWC = "NHWC"
DATA_FORMAT_NCHW = "NCHW"

class ConvDS(AbstractModule, Transposable):
    """
    This module is a convenience class that encapsulates a
    downsampling convolution + optionally GDN block:

    K x CONV -> (optionally) GDN

    transformation. If K > 1 then the downsampling will happen
    at the last convolution.

    It also implements its transpose for decoding, i.e. a

    K x DECONV -> (optionally) IGDN

    transformation.
    """

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 num_convolutions=1,
                 downsampling_rate=2,
                 padding="SAME",
                 use_gdn=True,
                 data_format=DATA_FORMAT_NHWC,
                 name="conv_gdn_block"):

        super(ConvDS, self).__init__(name=name)

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._num_convolutions = num_convolutions
        self._downsampling_rate = downsampling_rate
        self._padding = padding
        self._use_gdn = use_gdn
        self._data_format = data_format

    def _build(self, inputs):

        self._input_shape = tuple(inputs.get_shape().as_list())
        
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # Define first K - 1 convolutions
        self.pure_convs = [Conv2D(output_channels=self._output_channels,
                                  kernel_shape=self._kernel_shape,
                                  stride=1,
                                  padding=self._padding,
                                  data_format=self._data_format,
                                  name="conv{}".format(conv_idx))
                           for conv_idx in range(1, self._num_convolutions)]

        # Define downsampling convolution
        self.last_conv = Conv2D(output_channels=self._output_channels,
                                kernel_shape=self._kernel_shape,
                                stride=self._downsampling_rate,
                                padding=self._padding,
                                data_format=self._data_format,
                                name="conv{}".format(self._num_convolutions))


        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = inputs
        
        # Perform pure convolutions
        for conv in self.pure_convs:
            activations = conv(activations)

        # Downsampling convolution
        activations = self.last_conv(activations)

        # GDN activation
        if self._use_gdn:
            activations = tf.contrib.layers.gdn(activations,
                                                name="gdn")

        return activations


    def transpose(self, name=None):

        if name is None:
            name = self.module_name + "_transpose"

        def upsampling_output_shape():
            if self._data_format == DATA_FORMAT_NCHW:
                return self.last_conv.input_shape[2:4]
            else: # NHWC
                return self.last_conv.input_shape[1:3]

        def pure_output_shape():
            if self._data_format == DATA_FORMAT_NCHW:
                return self.pure_convs[0].input_shape[2:4]
            else: # NHWC
                return self.pure_convs[0].input_shape[1:3]

        return DeconvUS(output_channels=lambda: self.last_conv.input_channels,
                        upsampling_output_shape=upsampling_output_shape,
                        kernel_shape=self._kernel_shape,
                        num_deconvolutions=self._num_convolutions,
                        pure_output_shape=None if self._num_convolutions == 1 else pure_output_shape,
                        upsampling_rate=self._downsampling_rate,
                        padding=self._padding,
                        use_igdn=self._use_gdn,
                        data_format=self._data_format,
                        name=name)
    
    @property
    def input_shape(self):
        self._ensure_is_connected()
        
        return self._input_shape
        


class DeconvUS(AbstractModule, Transposable):
    """
    This module is a convenience class that encapsulates an
    upsampling deconvolution + optionally an IGDN block

    K x DECONV -> (optionally) IGDN

    transformation. If K > 1, the upsampling occurs on the first
    deconvolution.

    It also implements its transpose for encoding, i.e. a

    K x CONV -> (optionally) GDN

    transformation.
    """

    def __init__(self,
                 output_channels,
                 upsampling_output_shape,
                 kernel_shape,
                 num_deconvolutions=1,
                 pure_output_shape=None,
                 upsampling_rate=2,
                 padding="SAME",
                 use_igdn=True,
                 data_format=DATA_FORMAT_NHWC,
                 name="deconv_igdn_block"):

        super(DeconvUS, self).__init__(name=name)

        self._output_channels = output_channels
        self._upsampling_output_shape = upsampling_output_shape
        self._kernel_shape = kernel_shape
        self._num_deconvolutions = num_deconvolutions

        if num_deconvolutions > 1 and pure_output_shape is None:
            raise tf.errors.InvalidArgumentError(
                "If the number of deconvolutions is greater than 1, pure_output_shape must be specified!")

        self._upsampling_rate = upsampling_rate
        self._padding = padding
        self._use_igdn = use_igdn
        self._data_format = data_format

    def _build(self, inputs):

        self._input_shape = tuple(inputs.get_shape().as_list())
        
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First deconvolution
        self.first_deconv = Conv2DTranspose(output_channels=self._output_channels,
                                            output_shape=self._upsampling_output_shape,
                                            kernel_shape=self._kernel_shape,
                                            stride=self._upsampling_rate,
                                            padding=self._padding,
                                            data_format=self._data_format,
                                            name="deconv1")

        # Define last K - 1 deconvolutions
        pure_deconvs = [Conv2DTranspose(output_channels=self._output_channels,
                                        output_shape=self._pure_output_shape,
                                        kernel_shape=self._kernel_shape,
                                        stride=self._upsampling_rate,
                                        padding=self._padding,
                                        data_format=self._data_format,
                                        name="deconv{}".format(deconv_idx))
                        for deconv_idx in range(2, self._num_deconvolutions + 1)]

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = inputs
        
        if self._use_igdn:
            activations = tf.contrib.layers.gdn(activations,
                                                inverse=True,
                                                name="igdn")

        # Upsampling deconvolution
        activations = self.first_deconv(activations)

        # Pure deconvolutions
        for deconv in pure_deconvs:
            activations = deconv(activations)

        return activations


    def transpose(self, name=None):

        if name is None:
            name = self.module_name + "_transpose"

        return ConvDS(output_channels=lambda: self.first_deconv.input_channels,
                      kernel_shape=self._kernel_shape,
                      num_convolutions=self._num_deconvolutions,
                      downsampling_rate=self._upsampling_rate,
                      padding=self._padding,
                      use_gdn=self._use_igdn,
                      data_format=self._data_format,
                      name=name)

    @property
    def input_shape(self):
        self._ensure_is_connected()
        
        return self._input_shape
