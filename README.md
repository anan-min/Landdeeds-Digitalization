

**Overview**

This project involves converting the data in land deeds (ฉโนดที่ดิน) into digital form. The goal is to store both the text and the map drawings found in the land deeds using OCR (Optical Character Recognition).

Typhoon OCR is unable to fully extract both the map and text from the land deed, so additional techniques and tools are used, working as follows:

---

**Process for Data Extraction**

1. **Text Extraction**

   * Start by using Typhoon OCR to extract the text from the land deed image.
   * After OCR processing, the resulting image is sent along with features commonly found in land deeds, such as land registration code, district, subdistrict, and others.
   * The output from the LLM (Language Model) is then formatted into JSON.

2. **Image Extraction (Land Map)**

   * First, text within the image is detected using Tesseract OCR.
   * The image is then adjusted for contour detection.
   * After adjustment, the image is processed by computer vision (CV) to detect contours and extract the shape of the land as drawn on the deed.

---

**Challenges**

* **LLM and OCR Limitations**
  Despite supporting Thai language, OCR and LLM models still struggle with certain aspects, such as Thai handwriting and Thai numerals.

* **Map Extraction Issues**
  Using OCR and contour detection to extract the map can be problematic, as distinguishing between the map and unrelated lines or stamps (like compasses, stamps, and watermarks) on the deed is difficult.

* **Extraction of Scale and Compass Information**
  Land deeds often include details like the scale and compass directions. These might need to be extracted as well, depending on the user’s needs, but the approach may vary based on the application.

* **Using Object Detection for Text and Map**
  Replacing OCR with object detection may help clean the image better, as object detection does not rely on language. Additionally, using an advanced model might allow direct detection of the map. Further testing is needed to explore this option.

---

