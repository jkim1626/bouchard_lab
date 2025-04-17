import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_nmrml_peak_list(nmrml_file):
    try:
        # Read the file content
        with open(nmrml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all peak elements using regex
        # This approach is more robust for potentially malformed XML
        peaks = []
        
        # Pattern to match peak elements with their attributes
        peak_pattern = r'<peak\s+center="([^"]+)"\s+amplitude="([^"]+)"\s+width="([^"]+)"'
        
        # Find all matches
        matches = re.findall(peak_pattern, content)
        
        for match in matches:
            ppm = float(match[0])
            intensity = float(match[1])
            
            # Convert width to ppm (assuming width is in Hz and 800 MHz spectrometer)
            width_value = float(match[2])
            fwhm = width_value / 800000  # Convert Hz to ppm
            
            peaks.append({
                'ppm': ppm,
                'intensity': intensity,
                'fwhm': fwhm
            })
        
        return peaks
    
    except Exception as e:
        print(f"Error parsing nmrML file: {e}")
        return []

def simulate_spectrum(peaks, axis_ppm):
    # Initialize spectrum with zeros
    spectrum = np.zeros_like(axis_ppm)
    
    # Constant for converting FWHM to standard deviation
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Add each peak to the spectrum
    for peak in peaks:
        # Extract peak parameters
        ppm = peak['ppm']
        intensity = peak['intensity']
        fwhm = peak['fwhm']
        
        # Calculate standard deviation from FWHM
        sigma = fwhm * fwhm_to_sigma
        
        # Calculate Gaussian function for this peak
        gaussian = intensity * np.exp(-((axis_ppm - ppm)**2) / (2 * sigma**2))
        
        # Add to spectrum
        spectrum += gaussian
    
    return spectrum

def main():
    # Define the file path
    nmrml_file = "9.nmrML"
    spectrum_output_file = "9.txt"

    # Define the chemical shift axis (32768 points from 10 ppm to -2 ppm)
    axis_ppm = np.linspace(10, -2, 32768)
    
    # Parse the nmrML file to extract peak information
    peaks = parse_nmrml_peak_list(nmrml_file)
    
    if not peaks:
        print("No peaks found in the nmrML file.")
        return
    
    print(f"Found {len(peaks)} peaks in the nmrML file.")
    
    # Simulate the spectrum
    spectrum = simulate_spectrum(peaks, axis_ppm)
    
    # Plot the spectrum
    plt.figure(figsize=(12, 6))
    
    # Plot the spectrum (inverted x-axis for NMR convention)
    plt.plot(axis_ppm, spectrum)
    plt.gca().invert_xaxis()  # Invert x-axis for NMR convention
    
    # Add labels and title
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel('Intensity')
    plt.title('Simulated NMR Spectrum')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust y-axis to show the full spectrum
    plt.ylim(bottom=-1)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    

    # Save the spectrum as a row vector to a CSV file
    try:
        np.savetxt(spectrum_output_file, spectrum.reshape(1, -1), delimiter=",", fmt="%.10f")
        print(f"Spectrum saved as a row vector to {spectrum_output_file}")
    except Exception as e:
        print(f"Error saving spectrum to file: {e}")

    return spectrum 

if __name__ == "__main__":
    main()
