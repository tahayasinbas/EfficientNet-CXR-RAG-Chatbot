import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-semibold ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-medical-blue text-white hover:bg-medical-blue-700 shadow-medical hover:shadow-medical-strong focus-visible:ring-medical-blue-400",
        destructive:
          "bg-status-error text-white hover:bg-red-600 shadow-card hover:shadow-card-hover focus-visible:ring-status-error",
        outline:
          "border-2 border-border-default bg-surface text-text-primary hover:bg-bg-tertiary hover:border-medical-blue hover:text-medical-blue focus-visible:ring-medical-blue-400",
        secondary:
          "bg-bg-tertiary text-text-primary hover:bg-border-light shadow-card hover:shadow-card-hover focus-visible:ring-medical-blue-400",
        ghost:
          "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary",
        link:
          "text-medical-blue underline-offset-4 hover:underline",
        success:
          "bg-status-success text-white hover:bg-green-600 shadow-card hover:shadow-card-hover focus-visible:ring-status-success",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3 text-xs",
        lg: "h-12 rounded-lg px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

const Button = React.forwardRef(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
