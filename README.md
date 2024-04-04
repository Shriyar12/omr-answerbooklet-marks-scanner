 Automated OMR Image Processor and Result Generator

 This project streamlines the task of entering student marks by automatically extracting USN, subject code and question-wise marks from scanned answer booklet images. The extracted data is then automatically displayed on an html webpage and is also stored in an excel sheet automatically for further access.

Libraries used:
- OpenCV was used to preprocess the image and detect values of marked bubbles.
- Flask used to view data on a web page.
- Pandas library to write data in an excel sheet.
Kmeans for clustering the bubbles based on spatial distance to group the marked bubbles on the answer booklet and seperate usn, subject code and marks from eachother.
