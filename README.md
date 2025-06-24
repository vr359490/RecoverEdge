# RecoverEdge

RecoverEdge is a SwiftUI-based iOS application that generates personalized physical recovery routines and provides a comprehensive library of evidence-based recovery methods. Designed for athletes, fitness enthusiasts, and anyone seeking to optimize post-exercise recovery, RecoverEdge tailors routines based on available time, location, and equipment, and explains the science behind each method.

---

## Features

- **Personalized Recovery Plan Generator**:  
  Input your available time, select your location (home, gym, hotel, or outdoors), and specify what equipment you have. RecoverEdge instantly generates a recovery plan customized to your constraints and goals.

- **Curated Recovery Library**:  
  Browse a searchable collection of recovery methods, each with clear instructions, required equipment, time commitment, difficulty, and a summary of supporting research.

- **Scientific Rationale**:  
  Every method includes a “Why?” section detailing the research-backed benefits, so you can recover smarter and understand the physiological basis behind each technique.

- **Flexible & Accessible**:  
  Recovery plans adjust to your context—no equipment? Traveling? Short on time? RecoverEdge always offers actionable options.

---

## Example Recovery Methods

- **Legs Up The Wall**: Passive inversion to promote venous return and relaxation.
- **Foam Rolling**: Self-massage for muscle soreness reduction and improved mobility.
- **Deep Breathing**: Guided breathwork to reduce stress and activate the parasympathetic nervous system.
- **Cold Water Immersion**: Protocols for inflammation reduction and muscle recovery.
- **Gentle Stretching**: Static stretches for flexibility and muscle maintenance.
- **Massage Gun Therapy**: Percussive therapy to decrease delayed onset muscle soreness (DOMS).
- **Progressive Muscle Relaxation**: Mindful tension/release for total body relaxation.
- **Tennis Ball Foot Massage**: Plantar fascia release for runners and active individuals.

---

## How It Works

1. **Generate a Recovery Plan**
   - Select your available time (e.g., 15, 25, or 45 minutes).
   - Choose your location: Gym, Hotel, Home, or Outdoors.
   - Specify what equipment you have access to.
   - Tap "Generate Recovery Plan" to receive a sequenced, step-by-step routine.

2. **Explore the Recovery Library**
   - Browse all available recovery methods.
   - Tap the "Why?" button on any method to learn about its evidence base and benefits.

3. **Review Your Plan**
   - See each step with clear instructions, equipment needs, and time allocation.
   - Optionally, view the research behind every method in your plan.

---

## Data Model Overview

- **RecoveryMethod**:  
  - `name`: String  
  - `description`: String  
  - `duration`: Int (minutes)  
  - `equipment`: [String]  
  - `difficulty`: Int (1–3)  
  - `videoURL`: Optional String  
  - `researchInfo`: String  
  - `category`: String

- **RecoveryPlan**:  
  - `name`: String  
  - `totalDuration`: Int  
  - `methods`: [RecoveryMethod]  
  - `requiredEquipment`: [String]  
  - `suitableLocations`: [Location]

- **Location**:  
  - `.home`, `.gym`, `.hotel`, `.outdoors`  
  - Each with custom equipment options.

---

## Screenshots

<img width="347" alt="test screenshot" src="https://github.com/user-attachments/assets/9a8b6dda-9909-4f82-9df8-4c157fa5df2c" />


---

## Author

Victor Ruan  
Created: June 2025
