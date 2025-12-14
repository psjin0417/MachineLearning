import os
import matplotlib.pyplot as plt
import glob

# Remove Korean font settings and rely on default (English support)
# plt.rcParams['font.family'] = 'Malgun Gothic' 

def count_images(directory):
    if not os.path.exists(directory):
        return 0
    # Check for common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    count = 0
    for ext in extensions:
        count += len(glob.glob(os.path.join(directory, ext)))
    return count

def main():
    # Dataset paths - Augmented only
    dataset_dirs = {
        'Positive (Augmented)': './positive_aug',
        'Negative (Augmented)': './negative_aug'
    }

    # Count images
    counts = {}
    print("=== Dataset Status ===")
    for label, path in dataset_dirs.items():
        cnt = count_images(path)
        counts[label] = cnt
        print(f"{label}: {cnt}")

    names = list(counts.keys())
    values = list(counts.values())

    # [Settings] Bar Positions (Adjust these values to change bar locations)
    # The bars will be drawn at these X-coordinates.
    positions = [1.5, 3.5]  

    # Colors (Single color: skyblue)
    colors = 'skyblue'

    plt.figure(figsize=(8, 6))
    
    # Use 'positions' for x-axis coordinates
    bars = plt.bar(positions, values, color=colors, edgecolor='black', width=0.3)

    # Set x-axis labels at the specified positions
    plt.xticks(positions, names)

    # Display count on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height}', ha='center', va='bottom', fontweight='bold')

    # Adjust x-axis range to center the view around the positions
    # Example: Show from 0 to 5
    plt.xlim(0, 5)

    plt.title('Kickboard Dataset Distribution (Augmented Only)', fontsize=16, pad=20)
    plt.xlabel('Dataset Type', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
