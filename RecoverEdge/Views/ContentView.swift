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
            
            ChatView()
                .tabItem {
                    Image(systemName: "bubble.left")
                    Text("Chat")
                }
            
            SettingsListView()
                .tabItem {
                    Image(systemName: "list.bullet")
                    Text("Settings")
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
    @State private var customTime: String = ""
    @State private var showingCustomTime = false
    @State private var showingLocationSelector = false
    @State private var planToPresent: [RecoveryMethod]? = nil
    
     @State private var showingGuidedSession = false
     @State private var guidedSessionMethods: [RecoveryMethod] = []
     @State private var guidedSessionTime: Int = 0
    
    let timeOptions = [15, 25, 45]
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 30) {
                    // Header
                    VStack(spacing: 8) {
                        Text("How much time do you have?")
                            .font(.title2)
                            .fontWeight(.semibold)
                            .multilineTextAlignment(.center)
                        
                        Text("Choose your recovery session length")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.top, 20)
                    
                    //This section added from Claude
                    ResponsiveTimeSelectionView(
                        selectedTime: $selectedTime,
                        showingCustomTime: $showingCustomTime,
                        customTime: $customTime,
                        timeOptions: timeOptions
                    )
                    .frame(height: 300) // Fixed height instead of percentage
                    .padding(.horizontal, 16)
                    
                    Spacer()
                    
                    // Generate Button
                    Button(action: {
                        showingLocationSelector = true
                    }) {
                        
                        Text("Generate Recovery Plan")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(
                                Group {
                                    if canGenerate {
                                        LinearGradient(
                                            colors: [Color.brandTeal, Color.brandTealDark],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    } else {
                                        Color.gray
                                    }
                                }
                            )
                            .cornerRadius(12)
                    }
                    .disabled(!canGenerate)
                    .padding(.horizontal)
                    .padding(.bottom, 30)
                }
            }
            .navigationTitle("RecoverEdge")
            .navigationBarTitleDisplayMode(.inline)
        }

         .sheet(isPresented: $showingLocationSelector) {
             SimpleLocationSelectorView(
                 selectedTime: getTotalTime(),
                 onPlanGenerated: { plan in
                     planToPresent = plan
                     showingLocationSelector = false
                 }
             )
             .environmentObject(dataStore)
         }
         .sheet(item: Binding<PlanWrapper?>(
             get: { planToPresent.map { PlanWrapper(methods: $0, totalTime: getTotalTime()) } },
             set: { _ in planToPresent = nil }
         )) { planWrapper in
             GeneratedPlanView(
                 methods: planWrapper.methods,
                 totalTime: planWrapper.totalTime,
                 onStartGuidedSession: {
                     // Store the plan data for guided session
                     guidedSessionMethods = planWrapper.methods
                     guidedSessionTime = planWrapper.totalTime
                     
                     // Show guided session after a small delay
                     DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                         showingGuidedSession = true
                     }
                 }
             )
             .environmentObject(dataStore)
         }
         .fullScreenCover(isPresented: $showingGuidedSession) {
             GuidedRecoveryView(methods: guidedSessionMethods, totalTime: guidedSessionTime)
         }
    }
    
    private var canGenerate: Bool {
        if showingCustomTime {
            return !customTime.isEmpty && Int(customTime) != nil && Int(customTime)! > 0
        }
        return selectedTime > 0
    }
    
    private func getTotalTime() -> Int {
        if showingCustomTime, let custom = Int(customTime) {
            return custom
        }
        return selectedTime
    }
    
    // Replace the generateFinalPlan method with this updated version:
    private func generateFinalPlan(for location: Location) -> [RecoveryMethod] {
        let equipmentManager = EquipmentPreferencesManager.shared
        
        // Get saved equipment for the selected location
        let savedEquipment = Array(equipmentManager.getEquipment(for: location))
        
        // Generate plan using saved equipment preferences
        return generatePlan(selectedLocation: location, availableEquipment: savedEquipment)
    }
    
    private func generatePlan(selectedLocation: Location, availableEquipment: [String]) -> [RecoveryMethod] {
        let suitableMethods = dataStore.recoveryMethods.filter { method in
            if method.equipment.isEmpty {
                return true
            }
            return method.equipment.allSatisfy { requiredEquipment in
                availableEquipment.contains(requiredEquipment)
            }
        }
        
        var plan: [RecoveryMethod] = []
        var remainingTime = getTotalTime()
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
        
        return plan
    }
}

// MARK: - Location Selector for Plan Generation
struct SimpleLocationSelectorView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @Environment(\.presentationMode) var presentationMode
    @StateObject private var equipmentManager = EquipmentPreferencesManager.shared
    
    let selectedTime: Int
    let onPlanGenerated: ([RecoveryMethod]) -> Void
    
    @State private var selectedLocation: Location = .none
    @State private var isGenerating = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 30) {
                    // Header
                    VStack(spacing: 8) {
                        Text("Where will you recover?")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("Your \(selectedTime)-minute session")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 4)
                            .background(Color.brandTeal.opacity(0.1))
                            .cornerRadius(8)
                    }
                    .padding(.top, 20)
                    
                    // Location Selection with Equipment Preview
                    VStack(spacing: 16) {
                        ForEach(Location.allCases.filter { $0 != .none }, id: \.self) { location in
                            LocationCardView(
                                location: location,
                                isSelected: selectedLocation == location,
                                equipmentCount: equipmentManager.getEquipment(for: location).count,
                                onSelect: { selectedLocation = location }
                            )
                        }
                    }
                    .padding(.horizontal)
                    
                    Spacer()
                    
                    // Generate button
                    Button(action: generatePlan) {
                        HStack {
                            if isGenerating {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                                Text("Generating...")
                            } else {
                                Image(systemName: "sparkles")
                                Text("Generate Recovery Plan")
                            }
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            Group {
                                if selectedLocation != .none && !isGenerating {
                                    LinearGradient(
                                        colors: [Color.brandTeal, Color.brandTealDark],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                } else {
                                    Color.gray
                                }
                            }
                        )
                        .cornerRadius(12)
                    }
                    .disabled(selectedLocation == .none || isGenerating)
                    .padding(.horizontal)
                    .padding(.bottom, 30)
                }
            }
            .navigationTitle("Choose Location")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
    
    private func generatePlan() {
        isGenerating = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            let availableEquipment = Array(equipmentManager.getEquipment(for: selectedLocation))
            let plan = generateRecoveryPlan(availableEquipment: availableEquipment)
            onPlanGenerated(plan)
            isGenerating = false
        }
    }
    
    private func generateRecoveryPlan(availableEquipment: [String]) -> [RecoveryMethod] {
        // Filter methods based on available equipment
        let suitableMethods = dataStore.recoveryMethods.filter { method in
            if method.equipment.isEmpty {
                return true // No equipment required
            }
            return method.equipment.allSatisfy { requiredEquipment in
                availableEquipment.contains(requiredEquipment)
            }
        }
        
        var plan: [RecoveryMethod] = []
        var remainingTime = selectedTime
        var shuffledMethods = suitableMethods.shuffled()
        
        // Always include breathing if time allows
        if let breathingMethod = shuffledMethods.first(where: { $0.category == "Breathing" }),
           remainingTime >= breathingMethod.duration {
            plan.append(breathingMethod)
            remainingTime -= breathingMethod.duration
            shuffledMethods.removeAll { $0.id == breathingMethod.id }
        }
        
        // Add other methods based on remaining time
        for method in shuffledMethods {
            if remainingTime >= method.duration {
                plan.append(method)
                remainingTime -= method.duration
            }
            
            if remainingTime <= 2 { break } // Stop if very little time remains
        }
        
        // Fallback logic if no methods were selected
        if plan.isEmpty {
            // Try to add at least a breathing method since it requires no equipment
            if let breathingMethod = dataStore.recoveryMethods.first(where: { $0.category == "Breathing" }) {
                plan.append(breathingMethod)
            }
        }
        
        return plan
    }
}

// MARK: - Location Card View
struct LocationCardView: View {
    let location: Location
    let isSelected: Bool
    let equipmentCount: Int
    let onSelect: () -> Void
    
    var body: some View {
        Button(action: onSelect) {
            HStack {
                Image(systemName: iconForLocation(location))
                    .font(.title2)
                    .frame(width: 30)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(location.rawValue)
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    Text("\(equipmentCount) equipment items configured")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isSelected ? .brandTeal : .gray)
                    .font(.title3)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isSelected ? Color.brandTeal.opacity(0.1) : Color(.systemGray6))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? Color.brandTeal : Color.clear, lineWidth: 2)
            )
        }
    }
    
    private func iconForLocation(_ location: Location) -> String {
        switch location {
        case .gym: return "dumbbell"
        case .hotel: return "bed.double"
        case .home: return "house"
        case .court: return "sportscourt"
        case .none: return ""
        }
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
    
    // Add callback for when user wants to start guided session
    let onStartGuidedSession: () -> Void
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    
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
                        
                        // Get Started Button - Only show if there are methods
                        VStack(spacing: 16) {
                            Button(action: {
                                // Close this view and trigger guided session from parent
                                presentationMode.wrappedValue.dismiss()
                                onStartGuidedSession()
                            }) {
                                HStack(spacing: 12) {
                                    Image(systemName: "play.circle.fill")
                                        .font(.title2)
                                    Text("Get Started")
                                        .font(.headline)
                                        .fontWeight(.semibold)
                                }
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(
                                    LinearGradient(
                                        colors: [Color.brandTeal, Color.brandTealDark],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .cornerRadius(12)
                                .shadow(color: Color.brandTeal.opacity(0.3), radius: 8, x: 0, y: 4)
                            }
                            
                            Text("Start your guided recovery session")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
                        .padding(.top, 24)
                        .padding(.bottom, 40)
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
            
            // Equipment and category row
            HStack {
                // Category badge
                Text(method.category)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color.brandTeal.opacity(0.2))
                    .cornerRadius(4)
                
                // Equipment display
                if !method.equipment.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "wrench.adjustable")
                            .font(.caption2)
                            .foregroundColor(.orange)
                        
                        Text(formatEquipmentList(method.equipment))
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                } else {
                    HStack(spacing: 4) {
                        Image(systemName: "hand.raised.fill")
                            .font(.caption2)
                            .foregroundColor(.green)
                        
                        Text("No equipment")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                }
                
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
    
    // MARK: - Helper Method
    private func formatEquipmentList(_ equipment: [String]) -> String {
        switch equipment.count {
        case 0:
            return "None"
        case 1:
            return equipment[0]
        case 2:
            return equipment.joined(separator: ", ")
        case 3:
            return equipment.joined(separator: ", ")
        default:
            // For 4+ items, show first two and count
            let firstTwo = Array(equipment.prefix(2))
            let remaining = equipment.count - 2
            return "\(firstTwo.joined(separator: ", ")) +\(remaining) more"
        }
    }
}
