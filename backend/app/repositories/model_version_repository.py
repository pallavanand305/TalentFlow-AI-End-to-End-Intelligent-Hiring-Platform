"""Repository for model version data access"""

from typing import List, Optional
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.model_version import ModelVersion


class ModelVersionRepository:
    """Repository for model version operations"""
    
    def __init__(self):
        pass
    
    async def create(
        self,
        model_version: ModelVersion,
        db_session: AsyncSession
    ) -> ModelVersion:
        """Create a new model version"""
        db_session.add(model_version)
        await db_session.flush()
        await db_session.refresh(model_version)
        return model_version
    
    async def get_by_id(
        self,
        model_version_id: str,
        db_session: AsyncSession
    ) -> Optional[ModelVersion]:
        """Get model version by ID"""
        query = select(ModelVersion).where(ModelVersion.id == model_version_id)
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_model_name(
        self,
        model_name: str,
        db_session: AsyncSession
    ) -> List[ModelVersion]:
        """Get all versions for a specific model"""
        query = select(ModelVersion).where(
            ModelVersion.model_name == model_name
        ).order_by(ModelVersion.created_at.desc())
        
        result = await db_session.execute(query)
        return list(result.scalars().all())
    
    async def get_by_model_and_version(
        self,
        model_name: str,
        version: str,
        db_session: AsyncSession
    ) -> Optional[ModelVersion]:
        """Get specific model version"""
        query = select(ModelVersion).where(
            and_(
                ModelVersion.model_name == model_name,
                ModelVersion.version == version
            )
        )
        
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_stage(
        self,
        model_name: str,
        stage: str,
        db_session: AsyncSession
    ) -> List[ModelVersion]:
        """Get model versions by stage"""
        query = select(ModelVersion).where(
            and_(
                ModelVersion.model_name == model_name,
                ModelVersion.stage == stage
            )
        ).order_by(ModelVersion.created_at.desc())
        
        result = await db_session.execute(query)
        return list(result.scalars().all())
    
    async def get_latest_version(
        self,
        model_name: str,
        db_session: AsyncSession
    ) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        query = select(ModelVersion).where(
            ModelVersion.model_name == model_name
        ).order_by(ModelVersion.created_at.desc()).limit(1)
        
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_production_model(
        self,
        model_name: str,
        db_session: AsyncSession
    ) -> Optional[ModelVersion]:
        """Get the production version of a model"""
        query = select(ModelVersion).where(
            and_(
                ModelVersion.model_name == model_name,
                ModelVersion.stage == "Production"
            )
        ).order_by(ModelVersion.created_at.desc()).limit(1)
        
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def update_stage(
        self,
        model_name: str,
        version: str,
        new_stage: str,
        db_session: AsyncSession
    ) -> Optional[ModelVersion]:
        """Update the stage of a model version"""
        model_version = await self.get_by_model_and_version(
            model_name, version, db_session
        )
        
        if model_version:
            model_version.stage = new_stage
            await db_session.commit()
            await db_session.refresh(model_version)
        
        return model_version
    
    async def get_models_by_stage(
        self,
        stage: str,
        db_session: AsyncSession
    ) -> List[ModelVersion]:
        """Get all models in a specific stage"""
        query = select(ModelVersion).where(
            ModelVersion.stage == stage
        ).order_by(ModelVersion.model_name, ModelVersion.created_at.desc())
        
        result = await db_session.execute(query)
        return list(result.scalars().all())