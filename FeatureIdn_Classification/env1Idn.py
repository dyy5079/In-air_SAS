from ImgProcessing import findCenter, read_h5

# Example usage:
data = read_h5('Z:/PSU PhD/Projects/In-air_SAS/outputs/chip_h5/t3e2_06_ch1_chip1.h5')
center = findCenter(data['chip'], debug=True)
print("Detected O center (x, y):", center)




