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
            
//            RecoveryLibraryView()
//                .tabItem {
//                    Image(systemName: "books.vertical")
//                    Text("Library")
//                }
//                .environmentObject(dataStore)

        }
        .accentColor(Color.brandTeal2)
    }
}

struct SettingsRow: View {
    let iconName: String
    let title: String
    
    var body: some View {
        HStack {
            Image(systemName: iconName)
                .foregroundColor(.gray)
                .frame(width: 24, height: 24)
            
            Text(title)
                .foregroundColor(.primary)
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(.gray)
        }
        .padding(.vertical, 12)
        .padding(.horizontal)
    }
}

//struct SettingsListView: View {
//    var body: some View {
//        VStack(spacing: 0) {
//            NavigationLink(destination: RecoveryLibraryView()) {
//                SettingsRow(iconName: "laptopcomputer", title: "About")
//            }
//            Divider()
//            NavigationLink(destination: Text("Software Update Screen")) {
//                SettingsRow(iconName: "gearshape", title: "Software Update")
//            }
//            Divider()
//            NavigationLink(destination: Text("Storage Screen")) {
//                SettingsRow(iconName: "internaldrive", title: "Storage")
//            }
//        }
//        .background(
//            RoundedRectangle(cornerRadius: 12)
//                .fill(Color(UIColor.secondarySystemBackground))
//        )
//        .padding()
//    }
//}

struct SettingsListView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                NavigationLink(destination: RecoveryLibraryView().environmentObject(dataStore)) {
                    SettingsRow(iconName: "books.vertical", title: "Recovery Library")
                }
                
                
                Divider()
                //You can try 'dumbbell' for icon
                NavigationLink(destination: LocationEquipmentSelectorView2()) {
                    SettingsRow(iconName: "dumbbell", title: "Available Equipment")
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(UIColor.secondarySystemBackground))
            )
            .padding()
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

//MARK: Plan Generator View - Updated Flow
struct PlanGeneratorView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @State private var selectedTime: Int = 0
    @State private var customTime: String = ""
    @State private var showingCustomTime = false
    @State private var showingLocationSelector = false
    @State private var planToPresent: [RecoveryMethod]? = nil
    
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
            LocationEquipmentSelectorView(
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
            GeneratedPlanView(methods: planWrapper.methods, totalTime: planWrapper.totalTime)
                .environmentObject(dataStore)
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
}

// MARK: - New Location & Equipment Selector Sheet
struct LocationEquipmentSelectorView2: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @Environment(\.presentationMode) var presentationMode
    
//    let selectedTime: Int
//    let onPlanGenerated: ([RecoveryMethod]) -> Void
    
    @State private var selectedLocation: Location = .none
    @State private var selectedEquipmentNames: Set<String> = []
    @State private var isGenerating = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header with time context
//                    VStack(spacing: 8) {
//                        Text("Where will you recover?")
//                            .font(.title2)
//                            .fontWeight(.semibold)
//                    }
//                    .padding(.top, 10)
                    
                    // Location Selection
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Location")
                            .font(.headline)
                            .padding(.horizontal)
                        
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
                    
                    // Equipment Selection (only show if location is selected)
                    if selectedLocation != .none {
                        VStack(alignment: .leading, spacing: 16) {
                            VStack(alignment: .leading, spacing: 4) {
//                                Text("Available Equipment")
//                                    .font(.headline)
                                Text("Select what you have access to")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal)
                            
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
                        .transition(.asymmetric(
                            insertion: .move(edge: .top).combined(with: .opacity),
                            removal: .move(edge: .top).combined(with: .opacity)
                        ))
                    }
                    Spacer()
                }
            }
            //.navigationTitle("Step 2 of 2")
            //.navigationBarTitleDisplayMode(.inline)
            .navigationTitle("Available Equipment")
        }
        .animation(.easeInOut, value: selectedLocation)
        
    }
    
}

// MARK: - New Location & Equipment Selector Sheet
struct LocationEquipmentSelectorView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    @Environment(\.presentationMode) var presentationMode
    
    let selectedTime: Int
    let onPlanGenerated: ([RecoveryMethod]) -> Void
    
    @State private var selectedLocation: Location = .none
    @State private var selectedEquipmentNames: Set<String> = []
    @State private var isGenerating = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header with time context
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
                    .padding(.top, 10)
                    
                    // Location Selection
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Location")
                            .font(.headline)
                            .padding(.horizontal)
                        
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
                    
                    // Equipment Selection (only show if location is selected)
                    if selectedLocation != .none {
                        VStack(alignment: .leading, spacing: 16) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Available Equipment")
                                    .font(.headline)
                                Text("Select what you have access to")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal)
                            
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
//                        .transition(.opacity.combined(with: .scale(scale: 0.98)))
//                        .animation(.easeInOut, value: selectedLocation)
                        .transition(.asymmetric(
                            insertion: .move(edge: .top).combined(with: .opacity),
                            removal: .move(edge: .top).combined(with: .opacity)
                        ))
                    }
                    
                    Spacer()
                    
                    // Generate Final Plan Button
                    Button(action: generateFinalPlan) {
                        HStack {
                            if isGenerating {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                                Text("Generating...")
                            } else {
                                Image(systemName: "sparkles")
                                Text("Create My Recovery Plan")
                            }
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            Group {
                                if canGenerateFinalPlan && !isGenerating {
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
                    .disabled(!canGenerateFinalPlan || isGenerating)
                    .padding(.horizontal)
                    .padding(.bottom, 20)
                }
            }
            .navigationTitle("Step 2 of 2")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
        .animation(.easeInOut, value: selectedLocation)
    }
    
    private var canGenerateFinalPlan: Bool {
        selectedLocation != .none
    }
    
    private func generateFinalPlan() {
        isGenerating = true
        
        // Add small delay for better UX
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            let plan = generatePlan()
            onPlanGenerated(plan)
            isGenerating = false
        }
    }
    
    private func generatePlan() -> [RecoveryMethod] {
        // Your existing plan generation logic
        let availableEquipmentNames = Array(selectedEquipmentNames)
        
        let suitableMethods = dataStore.recoveryMethods.filter { method in
            if method.equipment.isEmpty {
                return true
            }
            return method.equipment.allSatisfy { requiredEquipment in
                availableEquipmentNames.contains(requiredEquipment)
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

// Helper wrapper for sheet presentation
struct PlanWrapper: Identifiable {
    let id = UUID()
    let methods: [RecoveryMethod]
    let totalTime: Int
}

// MARK: - Updated Generated Plan View
struct GeneratedPlanView: View {
    @EnvironmentObject var dataStore: RecoveryDataStore
    let methods: [RecoveryMethod]
    let totalTime: Int
    @Environment(\.presentationMode) var presentationMode
    @State private var showingResearch: RecoveryMethod?
    @State private var showingGuidedSession = false // New state for guided session
    
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
                                showingGuidedSession = true
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
        .fullScreenCover(isPresented: $showingGuidedSession) {
            GuidedRecoveryView(methods: methods, totalTime: totalTime)
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
