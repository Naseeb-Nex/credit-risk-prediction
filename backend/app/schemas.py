from pydantic import BaseModel, Field

class CreditRiskRequest(BaseModel):
    Age: int = Field(..., example=35)
    Sex: str = Field(..., example="male")
    Job: int = Field(..., example=2)
    Housing: str = Field(..., example="own")
    Saving_accounts: str | None = Field(None, example="little")
    Checking_account: str | None = Field(None, example="moderate")
    Credit_amount: int = Field(..., example=1200)
    Duration: int = Field(..., example=24)
    Purpose: str = Field(..., example="car")

class CreditRiskResponse(BaseModel):
    risk: str
    probability: float
