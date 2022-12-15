module.exports = (sequelize, DataTypes) => {
    return sequelize.define('SOUS_AGENCE', {
      id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      code_sous_agence: {
        type: DataTypes.STRING,

      },
      
      nom_sous_agence: {
        type: DataTypes.STRING,
      },
      addresse_sous_agence: {
        type: DataTypes.STRING,
      },
      city_sous_agence: {
        type: DataTypes.STRING,
      },
      country_sous_agence: {
        type: DataTypes.STRING,
      },
      phone_sous_agence: {
        type: DataTypes.STRING,
      },
      email_sous_agence: {
        type: DataTypes.STRING,
      }
       

    })
  }