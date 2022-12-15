module.exports = (sequelize, DataTypes) => {
    return sequelize.define('AGENCE', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      code_agence: {
        type: DataTypes.STRING,

      },
      
      nom_agence: {
        type: DataTypes.STRING,
      },
      statut_agence: {
        type: DataTypes.STRING,
      }
       

    })
  }