import streamlit as st
import torch
from your_agent_module import Agent, run_episode  # Import your agent-related modules

# Load the trained model
model_path = "model1.pth"
model = torch.load(model_path)

# Function to run an episode and calculate the score
def calculate_score():
    # Instantiate the agent with your trained model and device
    train_policy = Agent(model, device)

    # Run an episode with the agent
    score = run_episode(train_policy, show_progress=True, capture_video=True)
    
    return score

# Main function to run the Streamlit app
def main():
    st.title("Agent Score Prediction")

    # Button to calculate the score
    if st.button("Calculate Score"):
        # Calculate the score
        score = calculate_score()
        
        # Display the score
        st.write(f"Score: {score:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
