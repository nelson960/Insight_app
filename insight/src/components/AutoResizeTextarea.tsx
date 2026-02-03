import React from "react";

interface AutoResizeTextareaProps
    extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
    value: string;
}

export const AutoResizeTextarea = React.forwardRef<
    HTMLTextAreaElement,
    AutoResizeTextareaProps
>(({ value, className, style, ...props }, ref) => {
    return (
        <div
            className={className}
            style={{
                ...style,
                display: "grid",
                gridTemplateAreas: '"stack"',
                alignItems: "stretch",
                padding: 0,
                // Ensure the container inherits fonts to pass down
                font: "inherit",
                minHeight: "24px",
            }}
        >
            {/* Ghost element to prop open the height/width */}
            <div
                aria-hidden="true"
                style={{
                    gridArea: "stack",
                    visibility: "hidden",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    overflowWrap: "anywhere",
                    font: "inherit",
                    padding: 0,
                    margin: 0,
                    // Match textarea default sizing behavior
                    minHeight: "24px",
                }}
            >
                {value + " "}
            </div>

            {/* The actual textarea */}
            <textarea
                {...props}
                ref={ref}
                value={value}
                style={{
                    gridArea: "stack",
                    width: "100%",
                    height: "100%",
                    resize: "none",
                    overflow: "hidden",
                    font: "inherit",
                    background: "transparent",
                    border: "none",
                    outline: "none",
                    padding: 0,
                    margin: 0,
                    boxShadow: "none",
                }}
            />
        </div>
    );
});
