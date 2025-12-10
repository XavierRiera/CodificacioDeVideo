Our first approach to building the website was done entirely manually, without using any AI tools or CSS styling. Despite the simplicity, the website was fully functional.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/a5bfcc6d-63d7-49a7-814a-74df17307e6f" />

Next, we leveraged ChatGPT to implement a full suite of unit tests for the website. In this version, CSS was not separated; instead, it was included directly inside the HTML file.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/7f2cc0d3-0760-4c0b-8889-f05fff3024be" />

Finally, we used Claude to create a more complex design by implementing a separate CSS file (style.css). This approach enhanced the visual presentation and structure of the website.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/3a166eac-fe0d-48ca-a898-417024ecfd06" />
<img width="500" alt="image" src="https://github.com/user-attachments/assets/dbccf684-6532-40b0-9eb5-ef33152b7648" />

Despite the improvements, some functionalities did not work as expected:
- Displaying Information: When using the "Read Info" function, the displayed information appeared incorrectly formatted.
  
<img width="2558" height="496" alt="image" src="https://github.com/user-attachments/assets/5a2f350b-06e8-4dfb-a3a5-8e4b23fc7da3" />

- Downloading Videos: Video downloads were not functioning properly; the videos could not be downloaded correctly.
  
![Imatge de WhatsApp 2025-12-10 a les 13 05 15_81eaa60a](https://github.com/user-attachments/assets/d4462278-6390-450a-aae6-3cec8fedb25f)

- Number of Tracks Functionality: Some features, such as retrieving the number of tracks, were working correctly.
  
![Imatge de WhatsApp 2025-12-10 a les 13 05 50_fa29f267](https://github.com/user-attachments/assets/ecce0942-9bc6-41ce-99ab-129f8df4ccd0)

This demonstrates that while AI tools can help speed up development and handle repetitive tasks, relying on AI for more complex implementations does not always guarantee better results. In some cases, as we observed here, attempting to do more complicated tasks using AI without closely reviewing its output can introduce new problems that might have been avoided if we had done the work manually. Careful supervision and understanding of the AI-generated work remain essential to ensure functionality and correctness.
