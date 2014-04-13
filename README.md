#PISAT
piSat aims to implement a raspberry pi cubesat design to downlink high quality astrophotography data above the atmosphere.

 The project will use a blade-rf for high bandwidth data transmission and will use star tracking to give orbital position information as part of the dataset. This project is being run from the London branch of the event. 

This project is solving the PhoneSat: Convert Your Smartphone Into a Satellite challenge.

##Description
piSat aims to implement a raspberry pi cubesat design to downlink high quality astrophotography data above the atmosphere.

The project will use a blade-rf for high bandwidth data transmission and will use start tracking to give orbital position information as part of the dataset.

This project is being run from the London branch of the event.

Two main types of star tracking: 
1.Relative rotation and rate measurement 
2.The lost-in-space problem

The second problem is much more computationally difficult as you have to work out from the arrangement of stars in the sky, what you are looking at. There are quite a lot of papers available that discuss this problem but a lot of these are commercially protected algorithms that are licensed to specific companies.

For this project, we tried to use the openCV library and the Fast Library for Approximate Nearest Neighbour solving for feature matching between our raspberry pi camera image and our star database.

We used the HYG 2.0 database of stars and took those with a magnitude greater than 9. We rendered these into an image and used this as a comparison.

We tried out some small scale tests with success! Showing we could identify which area of the sky we had cropped an image from.

In reality, we are required to make the code more robust than this as our database of stars will not look exactly like the night sky. We do this by limiting the descriptor settings and SURF feature detection settings from OpenCV.

Additionally, we worked on the problem of high data rate transmission for cubesats. Typically, cubesats have simple radios and minimal downlink rates (1 baud or lower). We used the bladeRF software defined radio (SDR) to increase the data rates. This involved writing c code for taking raw samples from the SDR and analysing them to extract data.

We got this to work reasonably well to transmit short messages, however the analysis is not yet robust enough to send whole images as we would like to.

Overall, we gained a lot of knowledge of the OpenCV computer vision library and SDR radio processing.


##PROJECT INFORMATION

License: BSD 2-Clause "Simplified" or "FreeBSD" License (BSD-2-Clause)


