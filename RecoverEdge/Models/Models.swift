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
    case court = "Court"  // Changed from outdoors = "Outdoors"
    case none = ""

    var availableEquipment: [Equipment] {
        switch self {
        case .none: return []
        case .gym:
            return [
                Equipment(name: "Foam Roller", category: "Recovery"),
                Equipment(name: "Massage Gun", category: "Recovery"),
                Equipment(name: "Yoga Mat", category: "Mat"),
                Equipment(name: "Resistance Bands", category: "Bands"),
                Equipment(name: "Lacrosse Ball", category: "Recovery"),
                Equipment(name: "Stretching Straps", category: "Bands"),
                Equipment(name: "Ice Bath", category: "Temperature"),
                Equipment(name: "Sauna", category: "Temperature")
            ]
        case .hotel:
            return [
                Equipment(name: "Towel", category: "Basic"),
                Equipment(name: "Pillow", category: "Basic"),
                Equipment(name: "Wall", category: "Basic"),
                Equipment(name: "Chair", category: "Furniture"),
                Equipment(name: "Bathtub", category: "Water")
            ]
        case .home:
            return [
                Equipment(name: "Foam Roller", category: "Recovery"),
                Equipment(name: "Yoga Mat", category: "Mat"),
                Equipment(name: "Resistance Bands", category: "Bands"),
                Equipment(name: "Tennis Ball", category: "Recovery"),
                Equipment(name: "Towel", category: "Basic"),
                Equipment(name: "Wall", category: "Basic"),
                Equipment(name: "Stairs", category: "Structure"),
                Equipment(name: "Ice Pack", category: "Temperature")
            ]
        case .court:  // Changed from .outdoors
            return [
                Equipment(name: "Ground", category: "Basic"),
                Equipment(name: "Bench", category: "Furniture"),
                Equipment(name: "Water Bottle", category: "Basic"),
                Equipment(name: "Court Wall", category: "Structure"),  // More specific than just "Tree"
                Equipment(name: "Towel", category: "Basic")  // Added towel as courts often have them
            ]
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
        RecoveryMethod(
            name: "Legs Up The Wall",
            description: "Lie on your back with legs elevated against a wall",
            duration: 10,
            equipment: ["Wall"],
            difficulty: 1,
            videoURL: "676247342",
            researchInfo: "This passive inversion helps improve venous return, reducing swelling and promoting relaxation. Studies show it can help lower heart rate and activate the parasympathetic nervous system.",
            category: "Passive Recovery"
        ),
        RecoveryMethod(
            name: "Foam Rolling - Legs",
            description: "Roll out major leg muscle groups focusing on IT band, quads, and calves",
            duration: 8,
            equipment: ["Foam Roller"],
            difficulty: 2,
            videoURL: "676247342",
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
            name: "Gym Recovery Suite",
            totalDuration: 45,
            methods: [],
            requiredEquipment: ["Foam Roller", "Massage Gun", "Ice Bath"],
            suitableLocations: [.gym]
        ),
        // Optional: Add a court-specific plan
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
