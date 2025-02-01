
def single_spectrum(spectrum, freq_channels):
    import matplotlib.pyplot as plt
    for idx in range(spectrum.shape[0]):
        for jdx in range(spectrum.shape[1]):
            plt.plot(freq_channels, spectrum[idx,jdx])
    plt.xlim(min(freq_channels), max(freq_channels));

def verify_spectrum(test_spectrum, expected_spectrum, difference_spectrum, freq_channels):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=True, figsize=(7, 7))
    plt.subplots_adjust(wspace=0.4, hspace=0.2)

    for idx in range(expected_spectrum.shape[0]):
        for jdx in range(expected_spectrum.shape[1]):
            ax1.plot(freq_channels, expected_spectrum[idx,jdx])
    for idx in range(test_spectrum.shape[0]):
        for jdx in range(test_spectrum.shape[1]):    
            ax1.plot(freq_channels, test_spectrum[idx,jdx])
    for idx in range(difference_spectrum.shape[0]):
        for jdx in range(difference_spectrum.shape[1]): 
            ax2.plot(freq_channels, difference_spectrum[idx,jdx])

    fig.supxlabel("Frequency [GHz]")
    fig.suptitle("Verify Model Accuracy")

    ax1.set_title(f"Overlay Spectrums")
    ax1.set_ylabel(r'T$_b$ [K]')

    ax2.set_title(f"Percent Deviation (Mean = {np.mean(difference_spectrum):.2f}%)")
    ax2.set_ylabel('Deviation [%]')
    ax2.set_xlim(min(freq_channels), max(freq_channels));

