import SwiftUI

//// MARK: - Main App
//@main
//struct RecoverEdgeApp: App {
//    var body: some Scene {
//        WindowGroup {
//            ContentView()
//        }
//    }
//}

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

enum Location: String, CaseIterable {
    case gym = "Gym"
    case hotel = "Hotel"
    case home = "Home"
    case outdoors = "Outdoors"
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
        case .outdoors:
            return [
                Equipment(name: "Ground", category: "Basic"),
                Equipment(name: "Tree", category: "Nature"),
                Equipment(name: "Bench", category: "Furniture"),
                Equipment(name: "Water Bottle", category: "Basic")
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
            videoURL: nil,
            researchInfo: "This passive inversion helps improve venous return, reducing swelling and promoting relaxation. Studies show it can help lower heart rate and activate the parasympathetic nervous system.",
            category: "Passive Recovery"
        ),
        RecoveryMethod(
            name: "Foam Rolling - Legs",
            description: "Roll out major leg muscle groups focusing on IT band, quads, and calves",
            duration: 8,
            equipment: ["Foam Roller"],
            difficulty: 2,
            videoURL: nil,
            researchInfo: "Foam rolling helps break up fascial adhesions and improve blood flow. Research indicates it can reduce muscle soreness by 13% and improve range of motion.",
            category: "Self Massage"
        ),
        RecoveryMethod(
            name: "Deep Breathing",
            description: "Box breathing: 4 counts in, hold 4, out 4, hold 4",
            duration: 5,
            equipment: [],
            difficulty: 1,
            videoURL: nil,
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
    
    let curatedPlans: [RecoveryPlan] = [
        RecoveryPlan(
            name: "Quick Hotel Recovery",
            totalDuration: 15,
            methods: [], // Will be populated dynamically
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

// MARK: - Main Content View
struct ContentView: View {
    @StateObject private var dataStore = RecoveryDataStore()

    var body: some View {
        TabView {
            PlanGeneratorView()
                .tabItem {
                    Image(systemName: "wand.and.stars")
                    Text("Generate")
                }
                .environmentObject(dataStore)
            
            Text("Test Tab")
                .tabItem {
                    Image(systemName: "bubble.left")
                    Text("Chat")
                }
            
            RecoveryLibraryView()
                .tabItem {
                    Image(systemName: "books.vertical")
                    Text("Library")
                }
                .environmentObject(dataStore)

        }
        .accentColor(Color.brandTeal2)
    }
}

// MARK: - Plan Generator View
struct PlanGeneratorView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var selectedTime: Int = 0
    @State private var selectedLocation: Location = .none
    @State private var selectedEquipmentNames: Set<String> = []
    @State private var selectedEquipment: Set<Equipment> = []
    @State private var customTime: String = ""
    @State private var showingCustomTime = false
    @State private var generatedPlan: [RecoveryMethod] = []
    @State private var showingPlan = false
    
    let timeOptions = [15, 25, 45]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Time Selection
                    VStack(alignment: .leading, spacing: 12) {
//                        Text(" Available Time")
//                            .font(.headline)
                        
                        VStack(spacing: 12) {
                            HStack(spacing: 12) {
                                TimeButton(
                                    time: timeOptions[0], // "15 min"
                                    isSelected: selectedTime == timeOptions[0] && !showingCustomTime,
                                    action: {
                                        selectedTime = timeOptions[0]
                                        showingCustomTime = false
                                    }
                                )
                                
                                TimeButton(
                                    time: timeOptions[1], // "25 min"
                                    isSelected: selectedTime == timeOptions[1] && !showingCustomTime,
                                    action: {
                                        selectedTime = timeOptions[1]
                                        showingCustomTime = false
                                    }
                                )
                            }
                            
                            HStack(spacing: 12) {
                                TimeButton(
                                    time: timeOptions[2], // "45 min"
                                    isSelected: selectedTime == timeOptions[2] && !showingCustomTime,
                                    action: {
                                        selectedTime = timeOptions[2]
                                        showingCustomTime = false
                                    }
                                )
                                
                                Button(action: { showingCustomTime.toggle() }) {
                                    VStack{
                                        Image(systemName:"timer")
                                            .font(.system(size: 35))
                                        Text("CUSTOM")
                                        .   font(.system(size: 14, weight: .bold))
                                        
                                    }
                                        .padding(.horizontal, 30)
                                        .padding(.vertical, 30)
                                        
                                        .background(
                                            Group {
                                                if showingCustomTime {
                                                    LinearGradient(
                                                        colors: [Color.brandTeal, Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                                                        startPoint: .bottom,
                                                        endPoint: .top
                                                    )
                                                    
                                                } else {
                                                    LinearGradient(
                                                        colors: [Color(r:98, g:252, b:236),Color.brandTeal, Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                                                        startPoint: .bottom,
                                                        endPoint: .top
                                                    )
                                                }
                                            }
                                        )
                                        .foregroundColor(showingCustomTime ? .white : .primary)
                                        .cornerRadius(20)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity) // Force center alignment
                        .padding(.horizontal)
                        
                        if showingCustomTime {
                            TextField("Minutes", text: $customTime)
                                .keyboardType(.numberPad)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .padding(.horizontal)
                        }
                    }
                    
                    // Location Selection
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Location")
                            .font(.headline)
                        
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 12) {
                                ForEach(Location.allCases.filter { $0 != .none }, id: \.self) { location in
                                    LocationButton(
                                        location: location,
                                        isSelected: selectedLocation == location,
                                        action: {
                                            selectedLocation = location
                                            selectedEquipment = []
                                            selectedEquipmentNames.removeAll()
                                        }
                                    )
                                }
                            }
                            .padding(.horizontal)
                        }
                    }
                    
                    // Equipment Selection
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Available Equipment")
                            .font(.headline)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                            ForEach(selectedLocation.availableEquipment, id: \.id) { equipment in
                                EquipmentButton(
                                    equipment: equipment,
                                    isSelected: selectedEquipmentNames.contains(equipment.name),
                                    action: {
                                        if selectedEquipmentNames.contains(equipment.name) {
                                            selectedEquipmentNames.remove(equipment.name)
                                            selectedEquipment.remove(equipment)
                                        } else {
                                            selectedEquipmentNames.insert(equipment.name)
                                            selectedEquipment.insert(equipment)
                                        }
                                    }
                                )
                            }
                        }
                        .padding(.horizontal)
                    }
                    
                    // Generate Button
                    Button(action: generatePlan) {
                        Text("Generate Recovery Plan")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.black)
                            .cornerRadius(12)
                    }
                    .padding(.horizontal)
                    .padding(.top, 20)
                }
            }
            .navigationTitle("User Test's RecoverEdge")
        }
        .sheet(isPresented: $showingPlan) {
            GeneratedPlanView(methods: generatedPlan, totalTime: getTotalTime())
                .environmentObject(dataStore)
        }
    }
    
    private func getTotalTime() -> Int {
        if showingCustomTime, let custom = Int(customTime) {
            return custom
        }
        return selectedTime
    }
    
    private func generatePlan() {
        let targetTime = getTotalTime()
        let availableEquipmentNames = selectedEquipment.map { $0.name } + [""] // Include methods that need no equipment
        print("oogie boogie",availableEquipmentNames)
        
        let suitableMethods = dataStore.recoveryMethods.filter { method in
            method.equipment.isEmpty || method.equipment.allSatisfy { availableEquipmentNames.contains($0) }
        }
        
        var plan: [RecoveryMethod] = []
        var remainingTime = targetTime
        var shuffledMethods = suitableMethods.shuffled()
        
        // Always include breathing if time allows
        if let breathingMethod = shuffledMethods.first(where: { $0.category == "Breathing" }),
           remainingTime >= breathingMethod.duration {
            plan.append(breathingMethod)
            remainingTime -= breathingMethod.duration
            shuffledMethods.removeAll { $0.id == breathingMethod.id }
        }
        
        // Add other methods
        for method in shuffledMethods {
            if remainingTime >= method.duration {
                plan.append(method)
                remainingTime -= method.duration
            }
            
            if remainingTime <= 2 { break } // Stop if very little time left
        }
        
        // Add fallback if no equipment selected
        if plan.isEmpty || (selectedEquipment.isEmpty && selectedLocation == .hotel) {
            if let legsUpWall = dataStore.recoveryMethods.first(where: { $0.name == "Legs Up The Wall" }) {
                plan.append(legsUpWall)
            }
        }
        
        generatedPlan = plan
        showingPlan = true
    }
}

// MARK: - UI Components
struct TimeButton: View {
    let time: Int
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
                VStack(spacing: 0) {
                    Text("\(time)")
                        .font(.system(size: 28, weight: .bold))
                    Text("MIN")
                        .font(.system(size: 18, weight: .bold))
                }
                .multilineTextAlignment(.center)
                .padding(.horizontal, 39)
                .padding(.vertical, 30)
                .background(Group {
                    if isSelected {
                        LinearGradient(
                            colors: [Color.brandTeal, Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    } else {
                        LinearGradient(
                            colors: [Color(r:98, g:252, b:236),Color.brandTeal,Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    }
                })
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(20)
        }
    }
}

struct LocationButton: View {
    let location: Location
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: iconForLocation(location))
                    .font(.title2)
                Text(location.rawValue)
                    .font(.caption)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(
                Group {
                if isSelected {
                    LinearGradient(
                        colors: [Color.brandTeal2, Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                } else {
                    LinearGradient(
                        colors: [Color(r:98, g:252, b:236),Color.brandTeal,Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                }
            }
            )
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(12)
        }
    }
    
    private func iconForLocation(_ location: Location) -> String {
        switch location {
        case .gym: return "dumbbell"
        case .hotel: return "bed.double"
        case .home: return "house"
        case .outdoors: return "tree"
        case .none: return ""
        }
    }
}

struct EquipmentButton: View {
    let equipment: Equipment
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(equipment.name)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    Group {
                    if isSelected {
                        LinearGradient(
                            colors: [Color.brandTealDark,Color(r:30, g:80, b:80), Color.black],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    } else {
                        Color.gray.opacity(0.2)
                    }
                })
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(8)
        }
    }
}

// MARK: - Generated Plan View
struct GeneratedPlanView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    let methods: [RecoveryMethod]
    let totalTime: Int
    @Environment(\.presentationMode) var presentationMode
    @State private var showingResearch: RecoveryMethod?
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    // Plan Summary
                    VStack(spacing: 8) {
                        Text("Your Recovery Plan")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        Text("\(methods.count) methods • \(methods.reduce(0) { $0 + $1.duration }) minutes")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    
                    // Methods List
                    ForEach(Array(methods.enumerated()), id: \.element.id) { index, method in
                        MethodCard(
                            method: method,
                            stepNumber: index + 1,
                            showResearchAction: { showingResearch = method }
                        )
                    }
                    
                    Spacer(minLength: 100)
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
        .sheet(item: $showingResearch) { method in
            ResearchView(method: method)
        }
    }
}

struct MethodCard: View {
    let method: RecoveryMethod
    let stepNumber: Int
    let showResearchAction: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Step \(stepNumber)")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.brandTeal.opacity(0.2))
                    .cornerRadius(4)
                
                Spacer()
                
                Text("\(method.duration) min")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(method.name)
                .font(.headline)
            
            Text(method.description)
                .font(.body)
                .foregroundColor(.secondary)
            
            HStack {
                if !method.equipment.isEmpty {
                    Label("Equipment: \(method.equipment.joined(separator: ", "))", systemImage: "wrench.adjustable")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: showResearchAction) {
                    HStack(spacing: 4) {
                        Image(systemName: "questionmark.circle")
                        Text("Why?")
                    }
                    .font(.caption)
                    .foregroundColor(.black)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Research View
struct ResearchView: View {
    let method: RecoveryMethod
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(method.name)
                        .font(.title)
                        .fontWeight(.bold)
                    
                    Text("The Science")
                        .font(.headline)
                    
                    Text(method.researchInfo)
                        .font(.body)
                        .lineSpacing(4)
                    
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Duration")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(method.duration) minutes")
                                .font(.subheadline)
                                .fontWeight(.medium)
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .trailing) {
                            Text("Difficulty")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            HStack {
                                ForEach(1...3, id: \.self) { level in
                                    Circle()
                                        .fill(level <= method.difficulty ? Color.brandTeal : Color.gray.opacity(0.3))
                                        .frame(width: 8, height: 8)
                                }
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    
                    Spacer()
                }
                .padding()
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Recovery Library View

struct RecoveryLibraryView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var showingResearch: RecoveryMethod?
    
    var body: some View {
        NavigationView {
            List {
                Section("All Recovery Methods") {
                    ForEach(dataStore.recoveryMethods) { method in
                        MethodListRow(
                            method: method,
                            showResearchAction: { showingResearch = method }
                        )
                    }
                }
            }
            .navigationTitle("Recovery Library")
        }
        .sheet(item: $showingResearch) { method in
            ResearchView(method: method)
        }
    }
}

struct MethodListRow: View {
    let method: RecoveryMethod
    let showResearchAction: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(method.name)
                    .font(.headline)
                
                Spacer()
                
                Text("\(method.duration) min")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(method.description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .lineLimit(2)
            
            HStack {
                Text(method.category)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.brandTeal.opacity(0.2))
                    .cornerRadius(4)
                
                Spacer()
                
                Button(action: showResearchAction) {
                    HStack(spacing: 4) {
                        Image(systemName: "info.circle")
                        Text("Research")
                    }
                    .font(.caption)
                    .foregroundColor(.black)
                }
            }
        }
        .padding(.vertical, 4)
    }
}

//#Preview {
//    ContentView()
//}
