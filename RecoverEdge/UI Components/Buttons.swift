//
//  Buttons.swift
//  RecoverEdge
//
//  Created by Victor Ruan on 6/26/25.
//

import SwiftUI
import WebKit

// MARK: - UI Components
struct TimeButton: View {
    let time: Int
    let isSelected: Bool
    let action: () -> Void
    let buttonSize: CGSize // Add this parameter
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 0) {
                Text("\(time)")
                    .font(.system(size: scaledNumberFontSize, weight: .bold))
                Text("MIN")
                    .font(.system(size: scaledMinFontSize, weight: .bold))
            }
            .multilineTextAlignment(.center)
            .frame(width: buttonSize.width, height: buttonSize.height)
            .background(Group {
                if isSelected {
                    LinearGradient(
                        colors: [Color.brandTeal, Color.brandTealDark, Color(r:30, g:80, b:80), Color.black],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                } else {
                    LinearGradient(
                        colors: [Color(r:98, g:252, b:236), Color.brandTeal, Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                }
            })
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(scaledCornerRadius)
        }
    }
    
    // Computed properties for responsive sizing
    private var scaledNumberFontSize: CGFloat {
        // Scale the number font based on button size with safety checks
        let baseSize: CGFloat = 28
        let minButtonDimension = max(80, min(buttonSize.width, buttonSize.height)) // Ensure minimum
        let scaleFactor = minButtonDimension / 120 // Base size reference
        return 1.2*max(baseSize * scaleFactor, 16) // Minimum font size of 16
    }
    
    private var scaledMinFontSize: CGFloat {
        // Scale the "MIN" text proportionally
        return 1.2*max(scaledNumberFontSize * 0.64, 10) // Minimum font size of 10
    }
    
    private var scaledCornerRadius: CGFloat {
        // Scale corner radius based on button size with safety checks
        let minButtonDimension = max(80, min(buttonSize.width, buttonSize.height))
        return 1.2 * max(minButtonDimension * 0.15, 8) // Minimum corner radius of 8
    }
}

// MARK: - Updated Custom Time Button
struct CustomTimeButton: View {
    let isSelected: Bool
    let action: () -> Void
    let buttonSize: CGSize
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 0) {
                Image(systemName: "timer")
                    .font(.system(size: scaledIconSize))
                Text("CUSTOM")
                    .font(.system(size: scaledTextSize, weight: .bold))
            }
            .frame(width: buttonSize.width, height: buttonSize.height)
            .background(
                Group {
                    if isSelected {
                        LinearGradient(
                            colors: [Color.brandTeal, Color.brandTealDark, Color(r:30, g:80, b:80), Color.black],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    } else {
                        LinearGradient(
                            colors: [Color(r:98, g:252, b:236), Color.brandTeal, Color(r:80, g:190, b:190), Color(r:50, g:123, b:127)],
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    }
                }
            )
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(scaledCornerRadius)
        }
    }
    
    private var scaledIconSize: CGFloat {
        let baseSize: CGFloat = 35
        let minButtonDimension = max(80, min(buttonSize.width, buttonSize.height))
        let scaleFactor = minButtonDimension / 110
        return 1.2*max(baseSize * scaleFactor, 20)
    }
    
    private var scaledTextSize: CGFloat {
        let baseSize: CGFloat = 14
        let minButtonDimension = max(80, min(buttonSize.width, buttonSize.height))
        let scaleFactor = minButtonDimension / 100
        return 1.2*max(baseSize * scaleFactor, 10)
    }
    
    private var scaledCornerRadius: CGFloat {
        let minButtonDimension = max(80, min(buttonSize.width, buttonSize.height))
        return 1.2*max(minButtonDimension * 0.15, 8)
    }
}


// MARK: - Updated PlanGeneratorView Section
// Replace your current time selection section with this:

struct ResponsiveTimeSelectionView: View {
    @Binding var selectedTime: Int
    @Binding var showingCustomTime: Bool
    @Binding var customTime: String
    let timeOptions: [Int]
    
    var body: some View {
        GeometryReader { geometry in
            // Add safety checks and minimum values
            let safeWidth = max(geometry.size.width, 200) // Minimum width
            let safeHeight = max(geometry.size.height, 200) // Minimum height
            
            let availableWidth = safeWidth - 32 // Account for horizontal padding
            let availableHeight = safeHeight
            
            // Calculate button size with safety checks
            let spacing: CGFloat = 12
            let calculatedButtonWidth = (availableWidth - spacing) / 2
            let calculatedButtonHeight = (availableHeight - spacing) / 2
            
            // Ensure minimum button size and prevent negative values
            let minButtonSize: CGFloat = 80
            let maxButtonSize: CGFloat = 150
            
            let buttonWidth = max(minButtonSize, min(maxButtonSize, calculatedButtonWidth))
            let buttonHeight = max(minButtonSize, min(maxButtonSize, calculatedButtonHeight, buttonWidth))
            
            let buttonSize = CGSize(width: buttonWidth, height: buttonHeight)
            
            VStack(spacing: spacing) {
                HStack(spacing: spacing) {
                    TimeButton(
                        time: timeOptions[0],
                        isSelected: selectedTime == timeOptions[0] && !showingCustomTime,
                        action: {
                            selectedTime = timeOptions[0]
                            showingCustomTime = false
                        },
                        buttonSize: buttonSize
                    )
                    
                    TimeButton(
                        time: timeOptions[1],
                        isSelected: selectedTime == timeOptions[1] && !showingCustomTime,
                        action: {
                            selectedTime = timeOptions[1]
                            showingCustomTime = false
                        },
                        buttonSize: buttonSize
                    )
                }
                
                HStack(spacing: spacing) {
                    TimeButton(
                        time: timeOptions[2],
                        isSelected: selectedTime == timeOptions[2] && !showingCustomTime,
                        action: {
                            selectedTime = timeOptions[2]
                            showingCustomTime = false
                        },
                        buttonSize: buttonSize
                    )
                    
                    CustomTimeButton(
                        isSelected: showingCustomTime,
                        action: { showingCustomTime.toggle() },
                        buttonSize: buttonSize
                    )
                }
                
                if showingCustomTime {
                    TextField("Enter minutes", text: $customTime)
                        .keyboardType(.numberPad)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(maxWidth: max(120, availableWidth * 0.6))
                        .transition(.opacity.combined(with: .scale))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
        .frame(minHeight: 200) // Ensure minimum height for GeometryReader
        .animation(.easeInOut(duration: 0.3), value: showingCustomTime)
    }
}

// MARK: - Updated LocationButton
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
        case .court: return "sportscourt"  // Changed from "tree" to volleyball/sports court icon
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
