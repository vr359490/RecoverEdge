//
//  Models.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 6/26/25.
//

// This file contains data models to be used (structs, classes)

import SwiftUI
import WebKit

// MARK: - Data Models
struct Equipment: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let category: String

    // Consider equipments the same when they have the same name, not UUID

    // Custom equality based on name
    static func == (lhs: Equipment, rhs: Equipment) -> Bool {
        return lhs.name == rhs.name
    }

    // Custom hash based on name
    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}

// MARK: - Updated Location Enum
enum Location: String, CaseIterable {
    case gym = "Gym"
    case hotel = "Hotel"
    case home = "Home"
    case court = "Court"
    case none = ""

    // Static property containing all available equipment
    static let allAvailableEquipment: [Equipment] = [
        // Basic equipment
        Equipment(name: "Towel", category: "Basic"),
        Equipment(name: "Wall", category: "Basic"),
        Equipment(name: "Ground", category: "Basic"),
        Equipment(name: "Water Bottle", category: "Basic"),
        Equipment(name: "Pillow", category: "Basic"),
        
        // Furniture/Structure
        Equipment(name: "Chair", category: "Furniture"),
        Equipment(name: "Bench", category: "Furniture"),
        Equipment(name: "Stairs", category: "Structure"),
        Equipment(name: "Court Wall", category: "Structure"),
        
        // Mats and Bands
        Equipment(name: "Yoga Mat", category: "Mat"),
        Equipment(name: "Resistance Bands", category: "Bands"),
        Equipment(name: "Stretching Straps", category: "Bands"),
        
        // Recovery Equipment
        Equipment(name: "Foam Roller", category: "Recovery"),
        Equipment(name: "Massage Gun", category: "Recovery"),
        Equipment(name: "Lacrosse Ball", category: "Recovery"),
        Equipment(name: "Tennis Ball", category: "Recovery"),
        Equipment(name: "Ice Pack", category: "Temperature"),
        
        // Water-based
        Equipment(name: "Bathtub", category: "Water"),
        Equipment(name: "Ice Bath", category: "Temperature"),
        Equipment(name: "Cold Tank", category: "Temperature"),
        Equipment(name: "Hot Tank", category: "Temperature"),
        
        // Temperature Therapy
        Equipment(name: "Sauna", category: "Temperature"),
        Equipment(name: "Hot Pad", category: "Temperature"),
        
        // Advanced Equipment
        Equipment(name: "Red Light Therapy", category: "Light Therapy"),
        Equipment(name: "Normatec", category: "Compression"),
        Equipment(name: "Vibration Plate", category: "Vibration"),
        Equipment(name: "Hypervolt Gun", category: "Percussion")
    ]

    var availableEquipment: [Equipment] {
        switch self {
        case .none:
            return []
        case .gym, .hotel, .home, .court:
            // All locations now have access to all equipment
            return Location.allAvailableEquipment
        }
    }
}

struct RecoveryMethod: Identifiable {
    let id = UUID()
    let name: String
    let description: String
    let duration: Int // in minutes
    let equipment: [String]
    let difficulty: Int // 1-3
    let videoURL: String?
    let researchInfo: String
    let category: String
}

struct RecoveryPlan: Identifiable {
    let id = UUID()
    let name: String
    let totalDuration: Int
    let methods: [RecoveryMethod]
    let requiredEquipment: [String]
    let suitableLocations: [Location]
}

// MARK: - Data Store
class RecoveryDataStore: ObservableObject {
    let recoveryMethods: [RecoveryMethod] = [
        // MARK: - Original Recovery Methods
        RecoveryMethod(
            name: "Legs Up The Wall",
            description: "Lie on your back with legs elevated against a wall",
            duration: 10,
            equipment: ["Wall"],
            difficulty: 1,
            videoURL: "251521648",
            researchInfo: "This passive inversion helps improve venous return, reducing swelling and promoting relaxation. Studies show it can help lower heart rate and activate the parasympathetic nervous system.",
            category: "Passive Recovery"
        ),
        RecoveryMethod(
            name: "Foam Rolling - Legs",
            description: "Roll out major leg muscle groups focusing on IT band, quads, and calves",
            duration: 8,
            equipment: ["Foam Roller"],
            difficulty: 2,
            videoURL: "224710839",
            researchInfo: "Foam rolling helps break up fascial adhesions and improve blood flow. Research indicates it can reduce muscle soreness by 13% and improve range of motion.",
            category: "Self Massage"
        ),
        RecoveryMethod(
            name: "Deep Breathing",
            description: "Box breathing: 4 counts in, hold 4, out 4, hold 4",
            duration: 5,
            equipment: [],
            difficulty: 1,
            videoURL: "676247342",
            researchInfo: "Deep breathing activates the vagus nerve, triggering the body's relaxation response. Studies show it can reduce cortisol levels by up to 25%.",
            category: "Breathing"
        ),
        RecoveryMethod(
            name: "Cold Water Immersion",
            description: "Immerse in cold water (50-60°F) or take a cold shower",
            duration: 3,
            equipment: ["Bathtub", "Ice Bath"],
            difficulty: 3,
            videoURL: nil,
            researchInfo: "Cold exposure reduces inflammation and muscle damage. Research shows 11-15 minutes per week can increase norepinephrine by 200-300%.",
            category: "Temperature"
        ),
        RecoveryMethod(
            name: "Gentle Stretching",
            description: "Hold static stretches for major muscle groups",
            duration: 12,
            equipment: ["Yoga Mat"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Static stretching after exercise helps maintain flexibility and can reduce muscle stiffness. Best performed when muscles are warm.",
            category: "Stretching"
        ),
        RecoveryMethod(
            name: "Massage Gun Therapy",
            description: "Use massage gun on major muscle groups at medium intensity",
            duration: 6,
            equipment: ["Massage Gun"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Percussive therapy increases blood flow and can reduce delayed onset muscle soreness (DOMS) by up to 30% when used post-exercise.",
            category: "Self Massage"
        ),
        RecoveryMethod(
            name: "Progressive Muscle Relaxation",
            description: "Tense and release muscle groups starting from toes to head",
            duration: 15,
            equipment: [],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "PMR reduces muscle tension and stress hormones. Studies show it can improve sleep quality and reduce anxiety levels significantly.",
            category: "Relaxation"
        ),
        RecoveryMethod(
            name: "Tennis Ball Foot Massage",
            description: "Roll tennis ball under feet to release plantar fascia",
            duration: 4,
            equipment: ["Tennis Ball"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Plantar fascia release improves foot mobility and can reduce lower leg tension. Particularly beneficial for runners and athletes.",
            category: "Self Massage"
        ),
        
        // MARK: - New Recovery Methods with New Equipment
        
        // Red Light Therapy
        RecoveryMethod(
            name: "Red Light Recovery Session",
            description: "Expose targeted muscle groups to red light therapy for cellular recovery",
            duration: 15,
            equipment: ["Red Light Therapy"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Red light therapy (660-850nm) penetrates tissue to stimulate mitochondrial function and reduce inflammation. Studies show 20-40% reduction in muscle fatigue and improved recovery time.",
            category: "Light Therapy"
        ),
        RecoveryMethod(
            name: "Red Light Joint Therapy",
            description: "Focus red light on sore joints and areas of inflammation",
            duration: 12,
            equipment: ["Red Light Therapy"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Photobiomodulation therapy reduces joint inflammation and pain. Clinical studies demonstrate significant improvements in range of motion and reduced stiffness.",
            category: "Light Therapy"
        ),
        
        // Sauna
        RecoveryMethod(
            name: "Sauna Recovery Session",
            description: "Sit in sauna at 160-180°F for heat therapy and relaxation",
            duration: 20,
            equipment: ["Sauna"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Heat therapy increases blood flow, promotes muscle relaxation, and triggers heat shock proteins. Regular sauna use can improve cardiovascular health and reduce inflammation markers.",
            category: "Temperature"
        ),
        
        // Cold Tank
        RecoveryMethod(
            name: "Cold Tank Immersion",
            description: "Full body immersion in cold tank (45-55°F) for inflammation control",
            duration: 4,
            equipment: ["Cold Tank"],
            difficulty: 3,
            videoURL: nil,
            researchInfo: "Cold water immersion reduces muscle damage markers and inflammation. Studies show 10-24% reduction in muscle soreness when used within 1 hour post-exercise.",
            category: "Temperature"
        ),
        
        // Hot Tank
        RecoveryMethod(
            name: "Hot Tank Therapy",
            description: "Warm water immersion (98-104°F) for muscle relaxation and circulation",
            duration: 15,
            equipment: ["Hot Tank"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Warm water immersion increases blood flow and reduces muscle tension. The hydrostatic pressure aids lymphatic drainage and reduces swelling.",
            category: "Temperature"
        ),
        
        // Contrast Therapy
        RecoveryMethod(
            name: "Contrast Hot-Cold Therapy",
            description: "Alternate between hot tank (3 min) and cold tank (1 min) for 3 cycles",
            duration: 12,
            equipment: ["Hot Tank", "Cold Tank"],
            difficulty: 3,
            videoURL: nil,
            researchInfo: "Contrast therapy creates a vascular pump effect, alternately vasodilating and vasoconstricting blood vessels. This enhances circulation and waste product removal.",
            category: "Temperature"
        ),
        
        // Normatec
        RecoveryMethod(
            name: "Normatec Leg Recovery",
            description: "Full leg compression therapy using pneumatic compression",
            duration: 30,
            equipment: ["Normatec"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Pneumatic compression therapy enhances venous return and lymphatic drainage. Studies show 15-20% improvement in subsequent performance and reduced muscle soreness.",
            category: "Compression Therapy"
        ),
        RecoveryMethod(
            name: "Normatec Arm Recovery",
            description: "Upper body compression therapy for arms and shoulders",
            duration: 20,
            equipment: ["Normatec"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Upper body compression therapy improves circulation in arms and shoulders. Particularly effective for overhead athletes and those with upper body fatigue.",
            category: "Compression Therapy"
        ),
        
        // Vibration Plate
        RecoveryMethod(
            name: "Vibration Plate Recovery",
            description: "Stand on vibration plate with gentle movements for muscle activation",
            duration: 8,
            equipment: ["Vibration Plate"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Whole body vibration stimulates muscle spindles and improves neuromuscular control. Research shows improved flexibility and reduced muscle soreness.",
            category: "Active Recovery"
        ),
        RecoveryMethod(
            name: "Vibration Plate Stretching",
            description: "Perform static stretches while standing on vibration plate",
            duration: 10,
            equipment: ["Vibration Plate"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Vibration-assisted stretching can improve flexibility gains by 25-40% compared to static stretching alone through enhanced muscle relaxation.",
            category: "Active Recovery"
        ),
        
        // Hypervolt Gun
        RecoveryMethod(
            name: "Hypervolt Percussion Therapy",
            description: "Target specific muscle groups with high-frequency percussion",
            duration: 8,
            equipment: ["Hypervolt Gun"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "High-frequency percussion therapy penetrates deeper than traditional massage guns. Clinical studies show improved range of motion and reduced muscle stiffness.",
            category: "Percussion Therapy"
        ),
        
        // Hot Pad
        RecoveryMethod(
            name: "Targeted Heat Therapy",
            description: "Apply hot pad to sore muscles and joints for localized heat treatment",
            duration: 15,
            equipment: ["Hot Pad"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Localized heat therapy increases tissue temperature and blood flow. Effective for reducing muscle spasms and improving tissue elasticity before stretching.",
            category: "Temperature"
        ),
        
        // MARK: - Enhanced Stretching Methods
        
        // Shoulder Stretches
        RecoveryMethod(
            name: "Shoulder Mobility Routine",
            description: "Comprehensive shoulder stretches including cross-body and overhead reaches",
            duration: 8,
            equipment: [],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Regular shoulder mobility work prevents impingement and maintains healthy shoulder mechanics. Essential for overhead athletes and desk workers.",
            category: "Stretching"
        ),
        RecoveryMethod(
            name: "Band-Assisted Shoulder Stretches",
            description: "Use resistance bands for deeper shoulder and chest stretches",
            duration: 10,
            equipment: ["Resistance Bands"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Band-assisted stretching allows for greater range of motion and controlled tension. Studies show 15-20% greater flexibility improvements compared to unassisted stretching.",
            category: "Stretching"
        ),
        
        // Quad Stretches
        RecoveryMethod(
            name: "Quadriceps Stretch Routine",
            description: "Standing and lying quad stretches to release hip flexors and quads",
            duration: 6,
            equipment: [],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Quadriceps stretching improves hip mobility and can reduce lower back tension. Critical for runners and cyclists who develop tight hip flexors.",
            category: "Stretching"
        ),
        RecoveryMethod(
            name: "Band-Assisted Quad Stretches",
            description: "Use resistance bands for enhanced quadriceps and hip flexor stretching",
            duration: 8,
            equipment: ["Resistance Bands"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Band assistance allows for proper alignment and deeper stretches. Particularly effective for those with limited flexibility or balance issues.",
            category: "Stretching"
        ),
        
        // Calf Stretches
        RecoveryMethod(
            name: "Calf and Achilles Stretches",
            description: "Stretch gastrocnemius and soleus muscles using wall and floor stretches",
            duration: 6,
            equipment: ["Wall"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Calf stretching prevents Achilles tendon injuries and improves ankle mobility. Essential for runners and prevents plantar fasciitis.",
            category: "Stretching"
        ),
        RecoveryMethod(
            name: "Band-Assisted Calf Stretches",
            description: "Use resistance bands for seated and lying calf stretches",
            duration: 7,
            equipment: ["Resistance Bands"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Band-assisted calf stretching provides consistent tension and allows for progression. Effective for addressing calf tightness and improving dorsiflexion.",
            category: "Stretching"
        ),
        
        // Hamstring Stretches
        RecoveryMethod(
            name: "Hamstring Flexibility Routine",
            description: "Standing and seated hamstring stretches for posterior chain mobility",
            duration: 8,
            equipment: [],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Hamstring flexibility reduces lower back stress and improves hip hinge mechanics. Critical for preventing hamstring strains in athletes.",
            category: "Stretching"
        ),
        RecoveryMethod(
            name: "Band-Assisted Hamstring Stretches",
            description: "Use resistance bands for supine and seated hamstring stretches",
            duration: 10,
            equipment: ["Resistance Bands"],
            difficulty: 1,
            videoURL: nil,
            researchInfo: "Band-assisted hamstring stretching allows for relaxed positioning and gradual progression. Studies show improved hamstring length and reduced injury risk.",
            category: "Stretching"
        ),
        
        // Glute Stretches
        RecoveryMethod(
            name: "Glute and Hip Stretch Routine",
            description: "Figure-4 stretches and hip openers to release glute tension",
            duration: 8,
            equipment: [],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Glute stretching improves hip mobility and can reduce lower back pain. Essential for runners and those who sit for extended periods.",
            category: "Stretching"
        ),
        RecoveryMethod(
            name: "Band-Assisted Glute Stretches",
            description: "Use resistance bands for enhanced glute and piriformis stretches",
            duration: 10,
            equipment: ["Resistance Bands"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Band assistance helps achieve proper hip positioning and deeper glute stretches. Effective for addressing piriformis syndrome and hip impingement.",
            category: "Stretching"
        )
    ]

    // Update the curatedPlans array in RecoveryDataStore:
    let curatedPlans: [RecoveryPlan] = [
        RecoveryPlan(
            name: "Quick Hotel Recovery",
            totalDuration: 15,
            methods: [],
            requiredEquipment: ["Wall", "Towel"],
            suitableLocations: [.hotel]
        ),
        RecoveryPlan(
            name: "Complete Home Session",
            totalDuration: 25,
            methods: [],
            requiredEquipment: ["Foam Roller", "Yoga Mat"],
            suitableLocations: [.home]
        ),
        RecoveryPlan(
            name: "Premium Gym Recovery",
            totalDuration: 60,
            methods: [],
            requiredEquipment: ["Normatec", "Red Light Therapy", "Sauna", "Cold Tank"],
            suitableLocations: [.gym]
        ),
        RecoveryPlan(
            name: "Standard Gym Recovery",
            totalDuration: 45,
            methods: [],
            requiredEquipment: ["Foam Roller", "Massage Gun", "Vibration Plate"],
            suitableLocations: [.gym]
        ),
        RecoveryPlan(
            name: "Court Side Recovery",
            totalDuration: 12,
            methods: [],
            requiredEquipment: ["Water Bottle", "Towel"],
            suitableLocations: [.court]
        )
    ]
}

extension Color {
    init(r: Int, g: Int, b: Int) {
        self.init(
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255
        )
    }
    static let brandTeal = Color(r: 91, g: 225, b: 212)
    static let brandTeal2 = Color(r:80, g:190, b:190)
    static let brandTealDark = Color(r:50, g:123, b:127)
}
