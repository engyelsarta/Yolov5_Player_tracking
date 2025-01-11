import cv2
import torch
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set up Tkinter window
root = tk.Tk()
root.title("YOLOv5 Video Detection")

# Input field for player ID
player_id_entry = tk.Entry(root)
player_id_entry.pack()

# Label to show current ID
current_id_label = tk.Label(root, text="")
current_id_label.pack()

video_label = tk.Label(root)
video_label.pack()

# Load video
video_path = r"D:\Anatomy\Task2_AI_model\YOLO\videos\A1606b0e6_0 (16).mp4"
cap = cv2.VideoCapture(video_path)

players_ids = {}  # Dictionary to hold current player positions and IDs
next_id = 0  # Next available ID
threshold_distance = 10  # Minimum distance to consider it the same player
player_paths = {}  # To store the positions of each player for heatmap

# Load the empty field image
field_image_path = r"D:\Anatomy\Task2_AI_model\YOLO\dst.jpg"  # Replace with your field image path
field_image = cv2.imread(field_image_path)
field_image = cv2.resize(field_image, (640, 480))  # Resize to match video frame size


def update_frame():
    global players_ids, next_id
    ret, frame = cap.read()
    if not ret:
        return

    # Resize frame for performance
    frame = cv2.resize(frame, (640, 480))
    results = model(frame)  # Process frame
    boxes = results.xyxy[0].numpy()  # Get results

    current_players = {}
    player_centers = []

    # Draw bounding boxes and gather player centers
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        player_centers.append((int(center_x), int(center_y)))

        # Draw bounding box for detected player
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Track players
    for center in player_centers:
        assigned_id = None
        for player_pos, player_id in players_ids.items():
            distance = np.linalg.norm(np.array(center) - np.array(player_pos))
            if distance < threshold_distance:
                assigned_id = player_id
                break

        # If not assigned, give a new unique ID
        if assigned_id is None:
            assigned_id = next_id
            next_id += 1  # Increment next ID for future players

        current_players[center] = assigned_id

        # Track player positions for heatmap
        if assigned_id not in player_paths:
            player_paths[assigned_id] = []
        player_paths[assigned_id].append(center)

    # Get the player ID from the input field
    target_id = player_id_entry.get()

    # Draw player IDs on the frame based on input
    for center, player_id in current_players.items():
        if target_id == "" or str(player_id) == target_id:  # Show only the target player
            cv2.putText(frame, f"ID: {player_id}", (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (int(center[0] - 5), int(center[1] - 5)), (int(center[0] + 5), int(center[1] + 5)),
                          (255, 0, 0), -1)

    players_ids = current_players  # Update the players list

    # Convert frame to ImageTk
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(4, update_frame)  # Adjusted to 5 ms


def show_heatmap():
    target_id = player_id_entry.get()
    if target_id.isdigit() and int(target_id) in player_paths:
        positions = player_paths[int(target_id)]

        # Create a new Tkinter window for the heatmap
        heatmap_window = tk.Toplevel(root)
        heatmap_window.title(f'Heatmap for Player ID: {target_id}')

        # Create a label to display the heatmap
        heatmap_label = tk.Label(heatmap_window)
        heatmap_label.pack()

        # Function to update heatmap periodically
        def update_heatmap():
            if target_id.isdigit() and int(target_id) in player_paths:
                positions = player_paths[int(target_id)]

                # Create a blank image with the field background
                heatmap_image = field_image.copy()

                # heatmap_image = cv2.flip(heatmap_image, 0) #flip image verticall
                # heatmap_image = cv2.flip(heatmap_image, 0) #flip image horizontall
                # heatmap_image = cv2.flip(heatmap_image, 0) #flip image both => rot 180 deg

                # Initialize a heatmap (same size as the field image)
                heatmap = np.zeros_like(heatmap_image, dtype=np.float32)

                # Draw the player's path on the heatmap image
                for (x, y) in positions:
                    cv2.circle(heatmap, (x, y), 10, (0, 255, 0), -1)  # Draw larger circles to accumulate heat

                # Apply Gaussian blur to the heatmap to create the fading effect
                heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)  # Use a larger kernel for more blur effect

                # Normalize the heatmap to the range of 0 to 255
                heatmap = np.uint8(np.clip(heatmap, 0, 255))

                # Flip the heatmap vertically
                # heatmap = cv2.flip(heatmap, 0)  # Flip only the heatmap, not the field image
                # heatmap_image = cv2.flip(heatmap_image, 1)  #horizontall
                # heatmap_image = cv2.flip(heatmap_image, -1)  #both => rot 1180 deg

                # Apply color map to the heatmap
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Blend the flipped heatmap with the field image
                blended_image = cv2.addWeighted(heatmap_colored, 0.6, heatmap_image, 0.4, 0)

                # Convert blended image to ImageTk
                blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(blended_image_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                heatmap_label.imgtk = imgtk
                heatmap_label.configure(image=imgtk)

                # Schedule the heatmap to be updated again after 500ms
                heatmap_window.after(500, update_heatmap)
            else:
                print("Player ID not found or invalid.")

        # Start the periodic heatmap update
        update_heatmap()

    else:
        print("Player ID not found or invalid.")


def on_closing():
    cap.release()
    root.destroy()


# Button to show heatmap
heatmap_button = tk.Button(root, text="Show Heatmap", command=show_heatmap)
heatmap_button.pack()

root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()