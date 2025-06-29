import SwiftUI
import WebKit

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
            
//            Text("Test Tab")
//                .tabItem {
//                    Image(systemName: "bubble.left")
//                    Text("Chat")
//                }
            ChatView()
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

//MARK: Plan Generator View
struct PlanGeneratorView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var selectedTime: Int = 0
    @State private var selectedLocation: Location = .none
    @State private var selectedEquipmentNames: Set<String> = []
    @State private var customTime: String = ""
    @State private var showingCustomTime = false
    @State private var planToPresent: [RecoveryMethod]? = nil // Use optional to control sheet
    
    let timeOptions = [15, 25, 45]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Time Selection
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(spacing: 12) {
                            HStack(spacing: 12) {
                                TimeButton(
                                    time: timeOptions[0],
                                    isSelected: selectedTime == timeOptions[0] && !showingCustomTime,
                                    action: {
                                        selectedTime = timeOptions[0]
                                        showingCustomTime = false
                                    }
                                )
                                
                                TimeButton(
                                    time: timeOptions[1],
                                    isSelected: selectedTime == timeOptions[1] && !showingCustomTime,
                                    action: {
                                        selectedTime = timeOptions[1]
                                        showingCustomTime = false
                                    }
                                )
                            }
                            
                            HStack(spacing: 12) {
                                TimeButton(
                                    time: timeOptions[2],
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
                                            .font(.system(size: 14, weight: .bold))
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
                        .frame(maxWidth: .infinity)
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
                                        } else {
                                            selectedEquipmentNames.insert(equipment.name)
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
            .navigationTitle("Victor's RecoverEdge")
        }
        .sheet(item: Binding<PlanWrapper?>(
            get: { planToPresent.map { PlanWrapper(methods: $0, totalTime: getTotalTime()) } },
            set: { _ in planToPresent = nil }
        )) { planWrapper in
            GeneratedPlanView(methods: planWrapper.methods, totalTime: planWrapper.totalTime)
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
        
        // Create list of available equipment names
        var availableEquipmentNames = Array(selectedEquipmentNames)
        let locationEquipment = selectedLocation.availableEquipment.map { $0.name }
        availableEquipmentNames.append(contentsOf: locationEquipment)
        availableEquipmentNames = Array(Set(availableEquipmentNames))
        
        // Filter methods based on available equipment
        let suitableMethods = dataStore.recoveryMethods.filter { method in
            if method.equipment.isEmpty {
                return true
            }
            return method.equipment.allSatisfy { requiredEquipment in
                availableEquipmentNames.contains(requiredEquipment)
            }
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
            
            if remainingTime <= 2 { break }
        }
        
        // Fallback logic
        if plan.isEmpty && selectedLocation == .hotel {
            if let legsUpWall = dataStore.recoveryMethods.first(where: { $0.name == "Legs Up The Wall" }) {
                plan.append(legsUpWall)
            }
        }
        
        if plan.isEmpty {
            if let breathingMethod = dataStore.recoveryMethods.first(where: { $0.category == "Breathing" }) {
                plan.append(breathingMethod)
            }
        }
        
        
        // Set the plan to present - this will trigger the sheet
        planToPresent = plan
    }
}

// Helper wrapper for sheet presentation
struct PlanWrapper: Identifiable {
    let id = UUID()
    let methods: [RecoveryMethod]
    let totalTime: Int
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
                    // Debug information at the top
//                    VStack(spacing: 8) {
//                        Text("DEBUG INFO")
//                            .font(.caption)
//                            .foregroundColor(.red)
//                        Text("Methods count: \(methods.count)")
//                            .font(.caption)
//                        Text("Total time passed: \(totalTime)")
//                            .font(.caption)
//                        Text("Actual duration: \(methods.reduce(0) { $0 + $1.duration })")
//                            .font(.caption)
//
//                        if methods.isEmpty {
//                            Text("⚠️ NO METHODS IN ARRAY")
//                                .font(.caption)
//                                .foregroundColor(.red)
//                        }
//                    }
//                    .padding()
//                    .background(Color.yellow.opacity(0.3))
//                    .cornerRadius(8)
                    
                    // Plan Summary
                    VStack(spacing: 8) {
                        Text("Your Recovery Plan")
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        let actualDuration = methods.reduce(0) { $0 + $1.duration }
                        Text("\(methods.count) methods • \(actualDuration) minutes")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    
                    // Methods List
                    if methods.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "exclamationmark.triangle")
                                .font(.system(size: 50))
                                .foregroundColor(.orange)
                            
                            Text("No Methods Generated")
                                .font(.title2)
                                .fontWeight(.bold)
                            
                            Text("Try adjusting your time, location, or equipment selections and generate again.")
                                .multilineTextAlignment(.center)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                        }
                        .padding(.top, 50)
                    } else {
                        ForEach(Array(methods.enumerated()), id: \.element.id) { index, method in
                            MethodCard(
                                method: method,
                                stepNumber: index + 1,
                                showResearchAction: { showingResearch = method }
                            )
                        }
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
//        .onAppear {
//            print("GeneratedPlanView appeared with \(methods.count) methods")
//            for (index, method) in methods.enumerated() {
//                print("Method \(index + 1): \(method.name) - \(method.duration) min")
//            }
//        }
        .sheet(item: $showingResearch) { method in
            ResearchView(method: method)
        }
    }
}

struct MethodCard: View {
    let method: RecoveryMethod
    let stepNumber: Int
    let showResearchAction: () -> Void
    @State private var showingVideo = false
    @State private var isVideoLoading = true
    @State private var shouldAutoplay = false
    
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
            
            // Expandable Video Section with Autoplay
            if showingVideo {
                VStack(spacing: 8) {
                    // Video header
                    HStack {
                        HStack(spacing: 6) {
                            Image(systemName: "play.rectangle.fill")
                                .foregroundColor(.blue)
                            Text("Demonstration")
                                .font(.subheadline)
                                .fontWeight(.medium)
                        }
                        
                        Spacer()
                        
                    }
                    
                    // Video player with autoplay
                    ZStack {
                        Rectangle()
                            .fill(Color.black)
                            .frame(height: 200)
                            .cornerRadius(8)
                        
                        if isVideoLoading {
                            VStack(spacing: 8) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(1.2)
                                Text("Loading video...")
                                    .foregroundColor(.white)
                                    .font(.caption)
                            }
                        }
                        
                        if let vimeoID = method.videoURL, !vimeoID.isEmpty {
                            VimeoPlayerView(vimeoID: vimeoID, shouldAutoplay: shouldAutoplay)
                                .frame(height: 200)
                                .cornerRadius(8)
                                .opacity(isVideoLoading ? 0 : 1)
                                .onAppear {
                                    // Trigger autoplay after video appears
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                                        shouldAutoplay = true
                                        withAnimation(.easeInOut(duration: 0.5)) {
                                            isVideoLoading = false
                                        }
                                    }
                                }
                        }
                    }
                }
                .padding(.vertical, 4)
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .scale(scale: 0.95)).combined(with: .move(edge: .top)),
                    removal: .opacity.combined(with: .scale(scale: 0.95))
                ))
            }
            
            HStack {
                if !method.equipment.isEmpty {
                    Label("Equipment: \(method.equipment.joined(separator: ", "))", systemImage: "wrench.adjustable")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // Enhanced video button with play indication
                if let vimeoID = method.videoURL, !vimeoID.isEmpty {
                    Button(action: {
                        withAnimation(.spring(response: 0.6, dampingFraction: 0.7)) {
                            showingVideo.toggle()
                            if showingVideo {
                                isVideoLoading = true
                                shouldAutoplay = false // Reset autoplay state
                            } else {
                                shouldAutoplay = false // Stop autoplay when hiding
                            }
                        }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: showingVideo ? "stop.circle.fill" : "play.circle.fill")
                                .foregroundColor(showingVideo ? .red : .blue)
                            Text(showingVideo ? "Stop Video" : "Play Video")
                        }
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(showingVideo ? Color.red.opacity(0.1) : Color.blue.opacity(0.1))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(showingVideo ? Color.red.opacity(0.3) : Color.blue.opacity(0.3), lineWidth: 1)
                        )
                    }
                }
                
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
        .animation(.spring(response: 0.6, dampingFraction: 0.7), value: showingVideo)
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
