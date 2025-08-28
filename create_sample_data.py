import pandas as pd
import numpy as np
from pathlib import Path

# Create sample fake news dataset
def create_sample_dataset():
    # Sample real news headlines and content
    real_news = [
        {"title": "Scientists Discover New Species of Deep-Sea Fish", 
         "text": "Marine biologists have identified a new species of fish living in the deep ocean trenches. The discovery was made during a recent expedition to the Mariana Trench using advanced submersible technology."},
        {"title": "Local School Wins National Science Competition", 
         "text": "Students from Lincoln High School took first place in the national science fair with their innovative water purification system. The project aims to provide clean drinking water to rural communities."},
        {"title": "New Study Shows Benefits of Regular Exercise", 
         "text": "A comprehensive study involving 10,000 participants over 5 years demonstrates that regular physical activity significantly reduces the risk of cardiovascular disease and improves mental health."},
        {"title": "City Council Approves New Public Transportation Plan", 
         "text": "The city council unanimously voted to expand the public bus system, adding 15 new routes and electric buses to reduce emissions and improve accessibility for residents."},
        {"title": "Technology Company Announces Breakthrough in Solar Energy", 
         "text": "Researchers at SolarTech Inc. have developed a new type of solar panel that is 40% more efficient than current models, potentially revolutionizing renewable energy adoption."},
        {"title": "University Researchers Develop New Cancer Treatment", 
         "text": "A team of oncologists and biochemists have created a targeted therapy that shows promising results in early clinical trials for treating aggressive forms of lung cancer."},
        {"title": "Archaeological Team Uncovers Ancient Roman Settlement", 
         "text": "Excavations near the Italian countryside have revealed a well-preserved Roman villa dating back to the 2nd century AD, complete with intricate mosaics and artifacts."},
        {"title": "Climate Change Report Shows Rising Global Temperatures", 
         "text": "The latest IPCC report confirms that global average temperatures have risen by 1.2 degrees Celsius since pre-industrial times, emphasizing the urgent need for climate action."},
        {"title": "New Vaccine Shows 95% Effectiveness in Clinical Trials", 
         "text": "Phase 3 trials of a new influenza vaccine demonstrate exceptional efficacy rates, offering hope for better protection against seasonal flu outbreaks."},
        {"title": "Space Agency Successfully Launches Mars Rover Mission", 
         "text": "The latest Mars exploration rover was successfully launched today, equipped with advanced instruments to search for signs of past microbial life on the Red Planet."},
    ]
    
    # Sample fake news headlines and content
    fake_news = [
        {"title": "Aliens Land in Downtown Area, Government Covers Up", 
         "text": "Multiple witnesses claim to have seen extraterrestrial beings exit a UFO in the city center last night. Government officials deny the incident despite clear video evidence."},
        {"title": "Miracle Cure Discovered: Eating This Fruit Prevents All Diseases", 
         "text": "Local doctor claims that consuming dragon fruit daily can cure cancer, diabetes, and heart disease. Pharmaceutical companies are trying to suppress this information."},
        {"title": "Secret Society Controls World Economy Through Hidden Messages", 
         "text": "Investigative journalist uncovers evidence that a secret organization manipulates global markets using coded messages hidden in popular TV shows and movies."},
        {"title": "Scientists Prove Earth is Actually Flat Using New Technology", 
         "text": "A group of independent researchers using advanced laser technology have definitively proven that the Earth is flat, contradicting centuries of scientific consensus."},
        {"title": "Vaccines Contain Mind Control Chips, Whistleblower Reveals", 
         "text": "Former pharmaceutical employee exposes the truth about microchips in vaccines designed to control human thoughts and behavior through 5G networks."},
        {"title": "Time Traveler from 2050 Warns of Impending Disaster", 
         "text": "A man claiming to be from the future has provided detailed predictions about catastrophic events that will occur in the next decade, backed by impossible knowledge."},
        {"title": "Hidden Cure for Aging Found in Ancient Egyptian Tomb", 
         "text": "Archaeologists discover papyrus scrolls containing the secret to eternal youth, but the discovery is being suppressed by anti-aging industry corporations."},
        {"title": "Weather Control Technology Used to Create Natural Disasters", 
         "text": "Leaked documents reveal that governments possess weather manipulation technology and have been artificially creating hurricanes and earthquakes for political gain."},
        {"title": "Celebrity Death Hoax Exposed: Star Faked Own Death for Publicity", 
         "text": "Investigation reveals that the recent celebrity death was staged as an elaborate publicity stunt, with the star currently hiding in a secret location."},
        {"title": "Moon Landing Footage Confirmed as Hollywood Production", 
         "text": "Newly discovered evidence proves that the 1969 moon landing was filmed on a movie set, with detailed analysis of lighting and shadow inconsistencies."},
    ]
    
    # Create more samples by varying the existing ones
    all_real = []
    all_fake = []
    
    # Expand real news
    for i in range(50):
        base = real_news[i % len(real_news)]
        all_real.append({
            "title": base["title"],
            "text": base["text"] + f" Additional reporting confirms the accuracy of these findings through peer review and verification processes."
        })
    
    # Expand fake news  
    for i in range(50):
        base = fake_news[i % len(fake_news)]
        all_fake.append({
            "title": base["title"],
            "text": base["text"] + f" Mainstream media refuses to report on this story due to pressure from powerful interests."
        })
    
    # Create DataFrames
    real_df = pd.DataFrame(all_real)
    fake_df = pd.DataFrame(all_fake)
    
    return real_df, fake_df

def main():
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    real_df, fake_df = create_sample_dataset()
    
    # Save as separate files (matching prep.py expectations)
    real_df.to_csv("data/raw/True.csv", index=False)
    fake_df.to_csv("data/raw/Fake.csv", index=False)
    
    print(f"Created sample dataset:")
    print(f"Real news: {len(real_df)} samples")
    print(f"Fake news: {len(fake_df)} samples")
    print(f"Files saved to data/raw/")

if __name__ == "__main__":
    main()
