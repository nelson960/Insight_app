import { forwardRef } from "react";

type Props = {
  size?: number;
  color?: string;
  strokeWidth?: number;
  className?: string;
};

const SendUpIcon = forwardRef<SVGSVGElement, Props>(
  ({ size = 18, color = "currentColor", strokeWidth = 2, className = "" }, ref) => {
    return (
      <svg
        ref={ref}
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
      >
        <path d="M12 20V5" />
        <path d="M6 11l6-6 6 6" />
      </svg>
    );
  }
);

SendUpIcon.displayName = "SendUpIcon";

export default SendUpIcon;
